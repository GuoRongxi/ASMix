import random
import argparse
import numpy as np
import torch
from read_data import get_dataset
import torch.utils.data as Data
from model_armix import MixText
from sklearn.metrics import f1_score, precision_recall_fscore_support
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast as autocast


parser = argparse.ArgumentParser('MixText')
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--lrmain", default=0.00001, type=float, help="initial learning rate of bert model")
parser.add_argument("--lrlast", default=0.001, type=float, help="initial learning rate of MixText model")
parser.add_argument("--max_seq_len", default=150, type=int)
parser.add_argument("--mix_option", default=True, type=bool, help="if mix or not")
parser.add_argument("--model", type=str, default='bert-base-multilingual-cased', help="default == bert-base-chinese")
parser.add_argument("--mix_layers_set", nargs='+', default=[6, 7, 9, 12], type=int, help='numbers of layers which are supposed to mix')
parser.add_argument("--alpha", default=1, type=float, help="alpha param of beta distribution")
parser.add_argument("--dataset_filepath", type=str, default="../data/cade", help="path of file")
parser.add_argument('--save_filepath', default='../log/cade', type=str)
parser.add_argument('--comment', default='train', type=str)
parser.add_argument('--seed', default=56, type=int)

parser.add_argument('--temp_rate', default=1.0, type=float)
parser.add_argument('--smooth_rate', default=0.5, type=float)
parser.add_argument('--warmup_steps', default=100, type=int, help='Linear warmup over warmup_steps')

parser.add_argument("--mix_diff", default=False, type=bool, help="if mix_layers are different or not")
parser.add_argument("--use_amp", default=False, type=bool, help="if using amp to accelerate")
parser.add_argument('--trainfile', default='/processed_train.csv', type=str)


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # log
    train_log_name = args.save_filepath + '/' + args.comment + '_train.csv'
    val_log_name = args.save_filepath + '/' + args.comment + '_val.csv'

    train_log_file = open(train_log_name, 'wb')
    val_log_file = open(val_log_name, 'wb')

    train_df = pd.DataFrame(columns=['epoch', 'batch_idx', '1-loss','2-loss'])
    train_df.to_csv(train_log_file, header=True, index=False)

    val_df = pd.DataFrame(columns=['epoch', 'val loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1_macro_score',
                                   'val_weight_score', 'each_precision/category', 'each_recall/category',
                                   'each_f1score/category'])
    val_df.to_csv(val_log_file, header=True, index=False)

    train_dataset, test_dataset, n_labels, label_count, train_num = get_dataset(args.dataset_filepath, args.max_seq_len,
                                                                                args.trainfile, args.model)

    t_total = len(train_dataset)

    # set shuffle TRUE to avoid overfitting during training period , no need to set shuffle TRUE during testing period
    # but why batch_size is not consistent
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)

    model = MixText(n_labels, args.mix_option, model=args.model).to(device)
    smooth_model = BertForMaskedLM.from_pretrained(args.model).to(device)

    # optimizer = optim.AdamW(
    #     [
    #         {"params": model.module.bert.parameters(), "lr": args.lrmain},
    #         {"params": model.module.linear.parameters(), "lr": args.lrlast}
    #     ]
    # )

    if not args.use_amp:
        model = nn.DataParallel(model)
        smooth_model = nn.DataParallel(smooth_model)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer = optim.AdamW(
            [
                {"params": [p for n, p in model.module.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": args.lrmain, "weight_decay":0.01},
                {"params": [p for n, p in model.module.linear.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": args.lrlast, "weight_decay":0.01},
                {"params": [p for n, p in model.module.bert.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": args.lrmain, "weight_decay":0.0},
                {"params": [p for n, p in model.module.linear.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": args.lrlast, "weight_decay":0.0},
            ]
        )
    else:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer = optim.AdamW(
            [
                {"params": [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": args.lrmain, "weight_decay":0.01},
                {"params": [p for n, p in model.linear.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": args.lrlast, "weight_decay":0.01},
                {"params": [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": args.lrmain, "weight_decay":0.0},
                {"params": [p for n, p in model.linear.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": args.lrlast, "weight_decay":0.0},
            ]
        )

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    scheduler = None
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    train_criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train(train_loader, model, optimizer, scheduler, train_criterion, epoch, n_labels, train_log_file,
              smooth_model, args.temp_rate, args.smooth_rate, scaler, args.mix_diff, args.use_amp)
        predict_list, targets_list, test_loss, test_acc = validate(test_loader, model, criterion, epoch,
                                                                   mode='Test Stats')
        # precision and recall
        test_precision, test_recall, _, _ = precision_recall_fscore_support(targets_list, predict_list, average='macro')
        # f1
        test_macro_score = 2 * test_precision * test_recall / (test_precision + test_recall)
        # weighted f1
        test_weight_score = f1_score(targets_list, predict_list, average='weighted')
        # precision, recall, f1, support of each class, rather average of each class
        each_precision, each_recall, each_macro_score, each_support = precision_recall_fscore_support(targets_list,
                                                                                                      predict_list)

        print('each class : precision : {} , recall : {} , f1_score : {}'.format(each_precision, each_recall,
                                                                                 each_macro_score))
        print(
            'epoch : {} , test loss : {} , test acc : {} , test_precision : {} , test_recall : {} , test_macro_score : {} , '
            'test_weighted_score : {}'.format(epoch, test_loss, test_acc, test_precision, test_recall, test_macro_score,
                                              test_weight_score))

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


def smooth_embeds(smooth_model, inputs, temp_rate, smooth_rate, word_embeddings, device):
    input_probs = smooth_model(inputs)
    z = torch.zeros_like(input_probs[0])
    idx = inputs.reshape(inputs.shape[0], inputs.shape[1], 1).long()
    ont_hot = z.scatter_(2, idx, 1.0).to(device)
    smooth = torch.nn.functional.softmax(input_probs[0] / temp_rate, dim=-1).to(device)
    now_probs = smooth_rate * smooth + (1 - smooth_rate) * ont_hot
    inputs_embeds_smooth = now_probs @ word_embeddings.weight
    return inputs_embeds_smooth


def train(train_loader, model, optimizer, schedule, train_criterion, epoch, n_labels, train_log_file,
          smooth_model, temp_rate, smooth_rate, scaler, mix_diff, use_amp):
    # set batch_normalization and drop_out ON
    model.train()

    for batch_idx, (inputs, inputs2, targets, targets2) in enumerate(train_loader):
        inputs = inputs.to(device)
        inputs2 = inputs2.to(device)
        tmp_targets = targets.to(device)
        tmp_targets2 = targets2.to(device)

        mix_layer = np.random.choice(args.mix_layers_set, 1)[0] - 1
        if args.use_amp:
            with autocast():
                output, lbeta = model(input_ids=inputs, input_ids2=inputs2, lbeta=args.alpha, mix_layer=mix_layer)
                loss = lbeta * train_criterion(output, tmp_targets) + (1 - lbeta) * train_criterion(output, tmp_targets2)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output, lbeta = model(input_ids=inputs, input_ids2=inputs2, lbeta=args.alpha, mix_layer=mix_layer)
            loss = lbeta * train_criterion(output, tmp_targets) + (1 - lbeta) * train_criterion(output, tmp_targets2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if use_amp:
            word_embeddings = model.bert.get_input_embeddings().to(device)
        else:
            word_embeddings = model.module.bert.get_input_embeddings().to(device)

        inputs_embeds_smooth = smooth_embeds(smooth_model, inputs, temp_rate, smooth_rate, word_embeddings, device)
        inputs_embeds_smooth2 = smooth_embeds(smooth_model, inputs2, temp_rate, smooth_rate, word_embeddings, device)

        if mix_diff:
            mix_layer = np.random.choice(args.mix_layers_set, 1)[0] - 1
        
        if use_amp:
            with autocast():
                output, lbeta = model(inputs_embeds=inputs_embeds_smooth, inputs_embeds2=inputs_embeds_smooth2, lbeta=args.alpha, mix_layer=mix_layer)
                loss2 = lbeta * train_criterion(output, tmp_targets) + (1 - lbeta) * train_criterion(output, tmp_targets2)
        else:
            output, lbeta = model(inputs_embeds=inputs_embeds_smooth, inputs_embeds2=inputs_embeds_smooth2, lbeta=args.alpha, mix_layer=mix_layer)
            loss2 = lbeta * train_criterion(output, tmp_targets) + (1 - lbeta) * train_criterion(output, tmp_targets2)

        if batch_idx % 2000 == 0:
            print('epoch{}, step{}, 1-loss{}, 2-loss{}'.format(epoch, batch_idx, loss.item(), loss2.item()))
            tmp_data = pd.DataFrame([[epoch, batch_idx, loss.item(), loss2.item()]])
            tmp_data.to_csv(train_log_file, mode='a', header=False, index=False)        

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss2).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss2.backward()
            optimizer.step()


def validate(val_loader, model, criterion, epoch, mode):
    model.eval()

    with torch.no_grad():
        loss_total, samples_total, correct = 0, 0, 0
        predict_list, targets_list = [], []
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, lbeta = model(inputs)
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
    set_seed(args.seed)
    print(args.seed)
    main()
