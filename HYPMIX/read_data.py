from pytransformers import *
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import collections
import random
import torch

def get_dataset(filepath, max_seq_len, trainfile, model='bert_base_uncased'):
    tokenizer = BertTokenizer.from_pretrained(model)

    # train = np.array(pd.read_csv(filepath + "/processed_train.csv", header = None))
    train = np.array(pd.read_csv(filepath + trainfile, header = None))
    test = np.array(pd.read_csv(filepath + "/processed_test.csv", header = None))

    n_labels = len(set(train[:, 0]))
    train_num = len(train)

    label_count = dict(collections.Counter(train[:,0]).most_common())

    label_times = dict()
    max_count = list(label_count.values())[0]
    for key, count in label_count.items():
        label_times[key] = int(max_count / count)

    label_dict = dict()
    for label, count in label_count.items():
        label_dict[label] = [i for i,j in enumerate(train[:, 0]) if j == label]

    # select some raw train data pairs
    add_train = augment_data(train, label_dict, label_times, label_count)

    train_dataset = loader_train_data(add_train, tokenizer, max_seq_len)
    test_dataset = loader_test_data(test, tokenizer, max_seq_len)

    return train_dataset, test_dataset, n_labels, label_count, train_num



def augment_data(train, label_rows, label_times, label_count):
    """
    randomly select some raw data pairs
    ...but selection scheme is not understand...
    :param train: type : np.array ; size : [n, 2]
    :param label_rows:  type : dict(label, the list of lines which belong to this label)
    :param label_times: type : dict(label, max_count / count) , means unbalance degree
    :param label_count: type : dict(label, label_count)
    :return: add_train type : [[label1,text1],[label2,text2],...,[label_x,text_x]]  raw train data pairs
    """
    add_train = []
    max_count = list(label_count.values())[0]
    for label, rows in label_rows.items():
        for each_row in rows:
            add_train.append([train[each_row], train[each_row]])
        # label_times[label] = int(label_max_count / label_count) means lower max int
        # 1 <= label_max_count / label_count < 2
        if label_times[label] == 1:
            # this label is the one which owns max number of rows
            if max_count - label_count[label] == 0:
                continue
            # close but not the max_row_number label
            else:
                random.shuffle(rows)
                aug_data = rows[ : max_count - label_count[label]]
                for m in aug_data:
                    n = random.sample([k for k in range(0, len(train)) if k != m], 1)[0]
                    add_train.append([train[m], train[n]])
        else:
            for each_row in rows:
                choice_less = random.sample([k for k in range(0, len(train)) if k != each_row], label_times[label] - 1)
                for j in choice_less:
                    add_train.append([train[each_row], train[j]])
            if max_count - label_count[label] * label_times[label] > 0:
                random.shuffle(rows)
                aug_data = rows[ : max_count - label_count[label] * label_times[label]]
                for m in aug_data:
                    n = random.sample([k for k in range(0, len(train)) if k != m], 1)[0]
                    add_train.append(([train[m], train[n]]))
    random.shuffle(add_train)
    return add_train

class loader_train_data(Dataset):
    def __init__(self, add_train, tokenizer, max_seq_len):
        self.text = add_train
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text1 = self.text[idx][0][1]
        text2 = self.text[idx][1][1]
        label1 = self.text[idx][0][0]
        label2 = self.text[idx][1][0]
        text1_encode = self.get_tokenized(text1)
        text2_encode = self.get_tokenized(text2)
        return (torch.tensor(text1_encode), torch.tensor(text2_encode), torch.tensor(label1), torch.tensor(label2))

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]
        text_encode = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(text_encode))
        text_encode += padding
        return text_encode

class loader_test_data(Dataset):
    def __init__(self, data_text, tokenizer, max_seq_len):
        """

        :param data_text: type : np.array
        :param tokenizer:
        :param max_seq_len:
        """
        self.text = data_text[:, 1]
        self.labels = data_text[:, 0]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.text[idx]
        text_encode = self.get_tokenized(text)
        return (torch.tensor(text_encode), torch.tensor(self.labels[idx]))

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]
        text_encode = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(text_encode))
        text_encode += padding
        return text_encode

