import random
import argparse
import numpy as np
import torch
from read_data import get_dataset
import torch.utils.data as Data
from sklearn.metrics import f1_score, precision_recall_fscore_support
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from mixtext_poincare import MixText


parser = argparse.ArgumentParser('MixText')
parser.add_argument("--epochs", default = 50, type = int)
parser.add_argument("--batch_size", default = 8, type = int)
parser.add_argument("--lrmain", default = 0.00001, type = float, help = "initial learning rate of bert model")
parser.add_argument("--lrlast", default = 0.001, type = float, help = "initial learning rate of MixText model")
parser.add_argument("--max_seq_len", default = 50, type = int)
parser.add_argument("--mix_option", default = True, type = bool, help = "if mix or not")
parser.add_argument("--model", type = str, default = 'bert-base-uncased', help = "default == bert-base-chinese")
parser.add_argument("--mix_layers_set", nargs='+', default = [6,7,9,12], type=int, help='numbers of layers which are supposed to mix')
parser.add_argument("--alpha", default = 16, type=float, help = "alpha param of beta distribution")
parser.add_argument("--dataset_filepath", type = str, default="../data/THS", help="path of file")
parser.add_argument('--save_filepath', default='../log/THS', type=str)
parser.add_argument('--comment', default='HYPMIX', type=str)
parser.add_argument('--trainfile', default='/train_10%.csv', type=str)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


    train_dataset, test_dataset, n_labels, label_count, train_num = get_dataset(args.dataset_filepath, args.max_seq_len, args.trainfile, args.model)
    # set shuffle TRUE to avoid overfitting during training period , no need to set shuffle TRUE during testing period
    # but why batch_size is not consistent
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size = args.batch_size, shuffle = True)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size = 512, shuffle = False)

    model = MixText(n_labels, args.mix_option).cuda()
    model = nn.DataParallel(model)

    optimizer = optim.AdamW(
        [
            {"params" : model.module.bert.parameters(), "lr" : args.lrmain},
            {"params" : model.module.linear.parameters(), "lr" : args.lrlast}
        ]
    )

    scheduler = None
    train_criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train(train_loader, model, optimizer, scheduler, train_criterion, epoch, n_labels, train_log_file)
        predict_list, targets_list, test_loss, test_acc = validate(test_loader, model, criterion, epoch, mode='Test Stats')
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



def set_seed(seed):
    # pseudo random generator initial
    # one time random.seed for one time random.sample
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed could initial pseudo random generator on all devices (both GPU and CPU)
    torch.manual_seed(seed)


def train(train_loader, model, optimizer, schedule, train_criterion, epoch, n_labels, train_log_file):
    # set batch_normalization and drop_out ON
    model.train()

    for batch_idx, (inputs, inputs2, targets, targets2) in enumerate(train_loader):
        
        inputs = inputs.to(device)
        inputs2 = inputs2.to(device)
        tmp_targets = targets.to(device)
        tmp_targets2 = targets2.to(device)

        mix_layer = np.random.choice(args.mix_layers_set, 1)[0] - 1

        l = np.random.beta(args.alpha, args.alpha)

        logits = model(inputs, inputs2, l, mix_layer)

        loss = l * train_criterion(logits, tmp_targets) + (1 - l) * train_criterion(logits, tmp_targets2)


        if batch_idx % 2000 == 0:
            print('epoch{}, step{}, loss{}'.format(epoch, batch_idx, loss.item()))
            tmp_data = pd.DataFrame([[epoch, batch_idx, loss.item()]])
            tmp_data.to_csv(train_log_file, mode='a', header=False, index=False)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(val_loader, model, criterion, epoch, mode):
    model.eval()

    with torch.no_grad():
        loss_total, samples_total, correct = 0, 0, 0
        predict_list, targets_list = [], []
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs= model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)
            predicted, targets = np.array(predicted.cpu()), np.array(targets.cpu())
            predict_list.extend(list(predicted))
            targets_list.extend(list(targets))

            correct += (predicted == targets).sum()
            loss_total += loss * inputs.shape[0]
            samples_total += inputs.shape[0]

        acc_total = correct / samples_total
        loss_total = loss_total / samples_total

    return predict_list, targets_list, loss_total, acc_total





if __name__ == "__main__":
    set_seed(56)
    main()
