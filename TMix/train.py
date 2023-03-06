import argparse
import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from transformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset

from read_data import *
from mixtext import MixText
from sklearn.metrics import f1_score, precision_recall_fscore_support


parser = argparse.ArgumentParser(description='Pytorch MixText')
parser.add_argument('--data-path', type=str, default='../data/THS/', help='path to data folders')
parser.add_argument('--n-labeled', type=int, default=10, help='number of labeled data')
parser.add_argument('--unlabeled', type=int, default=5000, help='number of unlabeled data')
parser.add_argument('--model', type=str, default='bert-base-uncased', help='pretrained model')
parser.add_argument('--train-aug', type=bool, default=False, metavar='N', help='augment labeled training data')

parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
parser.add_argument('--batch_size_u', type=int, default=1, help='batch_size')

parser.add_argument('--mix-option', type=bool, default=True, help='mix option, whether to mix or not')

parser.add_argument('--lrmain', type=float, default=1e-5)
parser.add_argument('--lrlast', type=float, default=1e-3)

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--temp-change', type=int, default=1e6)
parser.add_argument('--T', type=float, default=0.5, help='temperature for sharpen function')

parser.add_argument('--val-iteration', type=int, default=413)
parser.add_argument('--co', type=bool, default=False)
parser.add_argument('--alpha', type=float, default=16)
parser.add_argument('--separate-mix', type=bool, default=True)
parser.add_argument('--mix-layers-set', type=list, default=[7, 9, 12])
parser.add_argument('--mix-method', type=int, default=0)
parser.add_argument('--margin', type=float, default=0.7)

parser.add_argument('--lambda-u', type=float, default=0)
parser.add_argument('--lambda-u-hinge', type=float, default=0)

parser.add_argument('--comment', default='TMix', type=str)
parser.add_argument('--save_filepath', default='../log/THS', type=str)

args = parser.parse_args()


best_acc = 0
total_steps = 0
flag = 0
print('Whether mix: ', args.mix_option)
print("Mix layers sets: ", args.mix_layers_set)


def main():

    # log
    train_log_name = args.save_filepath + '/' + args.comment + '_train.csv'
    val_log_name = args.save_filepath + '/' + args.comment + '_val.csv'

    train_log_file = open(train_log_name, 'wb')
    val_log_file = open(val_log_name, 'wb')

    train_df = pd.DataFrame(columns=['epoch', 'batch_idx', 'loss'])
    train_df.to_csv(train_log_file, header=True, index=False)

    val_df = pd.DataFrame(columns=['epoch', 'val loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1_macro_score',
                                   'val_weight_score', 'each_precision/category', 'each_recall/category',
                                   'each_f1score/category'])
    val_df.to_csv(val_log_file, header=True, index=False)


    global best_acc
    # Read dataset and build dataloaders
    train_labeled_set, test_set, n_labels = get_data(
        args.data_path, args.n_labeled, model=args.model, train_aug=args.train_aug)

    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)

    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=512, shuffle=False)

    # Define the model, set the optimizer
    model = MixText(n_labels, args.mix_option).cuda()
    model = nn.DataParallel(model)
    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.linear.parameters(), "lr": args.lrlast},
        ])

    scheduler = None

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()


    # Start training
    for epoch in range(args.epochs):

        train(labeled_trainloader, model, optimizer,
              scheduler, train_criterion, epoch, n_labels, args.train_aug)


        predict_list, targets_list, test_loss, test_acc = validate(
            test_loader, model, criterion, epoch, mode='Test Stats ')

        # precision and recall
        test_precision, test_recall, _, _ = precision_recall_fscore_support(targets_list, predict_list, average='macro')
        # f1
        test_macro_score = 2 * test_precision * test_recall / (test_precision + test_recall)
        # weighted f1
        test_weight_score = f1_score(targets_list, predict_list, average='weighted')
        # precision, recall, f1, support of each class, rather average of each class
        each_precision, each_recall, each_macro_score, each_support = precision_recall_fscore_support(targets_list, predict_list)


        print('each class : precision : {} , recall : {} , f1_score : {}'.format(each_precision, each_recall, each_macro_score))
        print('epoch : {} , test loss : {} , test acc : {} , test_precision : {} , test_recall : {} , test_macro_score : {} , '
              'test_weighted_score : {}'.format(epoch, test_loss, test_acc, test_precision, test_recall, test_macro_score, test_weight_score))

        tmp_list = [epoch, test_loss, test_acc, test_precision, test_recall, test_macro_score, test_weight_score,
                    each_precision, each_recall, each_macro_score]
        tmp_data = pd.DataFrame([tmp_list])
        tmp_data.to_csv(val_log_file, mode='a', header=False, index=False)

    train_log_file.close()
    val_log_file.close()



def train(labeled_trainloader, model, optimizer, scheduler, criterion, epoch, n_labels, train_aug=False):
    labeled_train_iter = iter(labeled_trainloader)
    model.train()

    global total_steps
    global flag
    if flag == 0 and total_steps > args.temp_change:
        print('Change T!')
        args.T = 0.9
        flag = 1

    for batch_idx in range(args.val_iteration):

        total_steps += 1


        try:
            inputs_x, targets_x, inputs_x_length = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, inputs_x_length = labeled_train_iter.next()


        batch_size = inputs_x.size(0)

        targets_x = torch.zeros(batch_size, n_labels).scatter_(
            1, targets_x.view(-1, 1), 1)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)


        mixed = 1

        l = np.random.beta(args.alpha, args.alpha)


        mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
        mix_layer = mix_layer - 1


        idx = torch.randperm(batch_size)


        input_a, input_b = inputs_x, inputs_x[idx]
        target_a, target_b = targets_x, targets_x[idx]
        length_a, length_b = inputs_x_length, inputs_x_length[idx]


        # Mix sentences' hidden representations
        logits = model(input_a, input_b, l, mix_layer)
        mixed_target = l * target_a + (1 - l) * target_b


        Lx = criterion(logits[:batch_size], mixed_target[:batch_size], epoch+batch_idx/args.val_iteration, mixed)

        loss = Lx

        #max_grad_norm = 1.0
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()


def validate(valloader, model, criterion, epoch, mode):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        predict_list, targets_list = [], []

        for batch_idx, (inputs, targets, length) in enumerate(valloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)
            predicted, targets = np.array(predicted.cpu()), np.array(targets.cpu())
            predict_list.extend(list(predicted))
            targets_list.extend(list(targets))

            correct += (predicted == targets).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        acc_total = correct/total_sample
        loss_total = loss_total/total_sample

    return predict_list, targets_list, loss_total, acc_total



class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, epoch, mixed=1):

        Lx = - \
            torch.mean(torch.sum(F.log_softmax(
                outputs_x, dim=1) * targets_x, dim=1))

        return Lx


if __name__ == '__main__':
    main()
