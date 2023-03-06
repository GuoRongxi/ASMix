import os
import argparse
import torch
from read_data_bert import *
import torch.utils.data as data
from model import VanillaBert
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_seq_len', default=100, type=int)
parser.add_argument('--lrbase', default=1e-5, type=float, help='learning rate for base bert model')
parser.add_argument('--lrlast', default=1e-3, type=float, help='learning rate for last classification MLP')
parser.add_argument('--model', default='bert-base-uncased', type=str)
parser.add_argument('--dataset_filepath', default='../data/THS', type=str)
parser.add_argument('--save_filepath', default='../log/THS', type=str)
parser.add_argument('--comment', default='AEDA', type=str)
parser.add_argument('--trainfile', default='/trainforAEDA10%.csv', type=str)
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

    train_dataset, val_dataset, n_labels = get_dataset(args.dataset_filepath, args.max_seq_len, args.trainfile)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    # model.to(device) is recommended rather than model.cuda() to load the model on gpu
    model = VanillaBert(n_labels, args.model).to(device)
    # nn.parellel.DistributedDataParellel is recommanded, wait to change...
    model = nn.DataParallel(model)

    optimizer = optim.AdamW(
        [
            {'params': model.module.bert.parameters(), 'lr': args.lrbase},
            {'params': model.module.MLP.parameters(), 'lr': args.lrlast}
        ]
    )
    # CE is useful for classification
    criterion = nn.CrossEntropyLoss()
    best_epoch, best_precision, best_recall, best_macro_score, best_weight_score = 0, 0, 0, 0, 0
    each_precision, each_recall, each_macro_score, each_support = 0, 0, 0, 0

    for epoch in range(args.epochs):
        # train for one epoch
        train(train_loader, model, optimizer, criterion, epoch, train_log_file)
        # gain the loss of the current epoch on the whole val dataset
        # good for watching whether overfitting
        val_loss, predict_list, targets_list, val_acc = validate(val_loader, model, criterion, epoch)

        # set the parameter average 'macro'
        # calculate metrics for each label, and find their unweighted mean.
        # do not considerate label imbalance
        val_precision, val_recall, val_f1_macro_score, _ = precision_recall_fscore_support(targets_list, predict_list,
                                                                                           average='macro')
        val_weight_score = f1_score(targets_list, predict_list, average='weighted')

        # parameter 'average' is None, thus return an array for per metric
        each_precision, each_recall, each_f1score, each_support = precision_recall_fscore_support(targets_list,
                                                                                                  predict_list)

        print('precision recall f1_score of each class:')
        print(each_precision, each_recall, each_f1score)
        print('epoch:{},val loss{},val_acc{},val_precision{},val_recall{},val_f1_score{},val_weight_score{}'.format(
            epoch, val_loss, val_acc, val_precision, val_recall, val_f1_macro_score, val_weight_score
        ))

        tmp_list = [epoch, val_loss, val_acc, val_precision, val_recall, val_f1_macro_score, val_weight_score,
                    each_precision, each_recall, each_f1score]
        tmp_data = pd.DataFrame([tmp_list])
        tmp_data.to_csv(val_log_file, mode='a', header=False, index=False)

    train_log_file.close()
    val_log_file.close()


def train(train_loader, model, optimizer, criterion, epoch, train_log_file):
    """
    call train func once means executing for one epoch
    :param train_loader:  dataset loader
    :param model: bert
    :param optimizer: AdamW
    :param criterion: CELoss for classification
    :param epoch: just for print
    :return:
    """
    # indicate the model is during the period of training and enable Dropout module
    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        if batch_idx % 2000 == 0:
            print('epoch {}, step {}, train loss {}'.format(epoch, batch_idx, loss.item()))
            tmp_data = pd.DataFrame([[epoch, batch_idx, loss.item()]])
            tmp_data.to_csv(train_log_file, mode='a', header=False, index=False)
        loss.backward()
        optimizer.step()


def validate(val_loader, model, criterion, epoch):
    """
    use all data on the current model
    :param val_loader: dataset loader
    :param model: bert
    :param criterion: CELoss
    :param epoch: no use
    :return: metrics like precision, recall and f1score..
    """
    # indicate the model is during the period of validating and disable Dropout module
    model.eval()

    # not tracking history
    with torch.no_grad():
        loss_total, total_sample, correct = 0, 0, 0
        predict_list, targets_list = [], []
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # CELoss :
            # shape of outputs : (batch_size, class_num)
            # shape of targets : (batch_size)
            loss = criterion(outputs, targets)
            # shape of outputs is like tensor[[0 0 1 0]
            #                                 [1 0 0 0 ]
            #                                 [0 1 0 0]]
            # which number of classes is specified to 4
            # get the predicted class at dim 1 for per data X
            _, predicted = torch.max(outputs, 1)
            predicted, targets = np.array(predicted.cpu()), np.array(targets.cpu())
            predict_list.extend(list(predicted))
            targets_list.extend(list(targets))
            # boolean ndarray to sum to calculate number of True
            correct += (predicted == targets).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        acc_total = correct / total_sample
        loss_total = loss_total / total_sample
    return loss_total, predict_list, targets_list, acc_total


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seed(56)
    main()
