import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pytransformers import *
import collections
from eda_English import *
# from eda import *
import random

def get_dataset(filepath,max_seq_len,trainfile,model='bert-base-uncased'):

    tokenzier = BertTokenizer.from_pretrained(model)

    # train = np.array (pd.read_csv (filepath + "/processed_train.csv", header=None))
    train = np.array (pd.read_csv (filepath + trainfile, header=None))
    # val = np.array(pd.read_csv(r"./data/r8/val.csv", header=None))
    test = np.array (pd.read_csv (filepath + "/processed_test.csv", header=None))
    n_labels = len(set(train[:, 0]))
    label_count = dict (collections.Counter (train[:, 0]).most_common ())
    #print (label_count)
    label_dict = {}.fromkeys (label_count.keys (), 0)
    label_times = dict ()
    for key, count in label_count.items ():
        label_times[key] = int (list (label_count.values ())[0] / count)
    for label, count in label_count.items ():
        label_dict[label] = [i for i, j in enumerate (train[:, 0]) if j == label]
    # less_class = dict ([(key, value) for key, value in count_dict.items () if float (value) >= 4.0])
    # many_class = dict ([(key, value) for key, value in count_dict.items () if float (value) < 4.0])
    # print (many_class)
    # print (less_class)

    add_train = []
    alpha_sr, alpha_ri, alpha_rs, alpha_rd = 0.1, 0.1, 0.1, 0.1
    for key, temp in label_dict.items ():
        if label_times[key] == 1:
            if list (label_count.values ())[0] - label_count[key] == 0:
                continue
            else:
                random.shuffle (temp)
                aug_data = temp[:list (label_count.values ())[0] - label_count[key]]
                for m in aug_data:
                    # [k for k in range (0, len (train)) if k != m]
                    aug_line = gen_eda(train[m], alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=1)
                    add_train.extend(aug_line)
        else:
            for i in temp:
                # [k for k in range (0, len (train)) if k != i]
                aug_data_line = gen_eda (train[i], alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=label_times[key] - 1)
                add_train.extend (aug_data_line)
            if list (label_count.values ())[0] - label_count[key] * label_times[key] > 0:
                random.shuffle (temp)
                aug_data = temp[:list (label_count.values ())[0] - label_count[key] * label_times[key]]
                for m in aug_data:
                    aug_line = gen_eda (train[m], alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=1)
                    add_train.extend (aug_line)
    add_train = add_train + list(train)
    random.shuffle (add_train)
    train = np.array(add_train)
    #print (len (add_train))
    label_count = dict (collections.Counter (train[:, 0]).most_common ())
    #print (label_count)

    train_dataset = loader_data(train, tokenzier, max_seq_len)
    # val_dataset = loader_data(val, tokenzier, max_seq_len)
    test_dataset = loader_data(test, tokenzier, max_seq_len)
    return  train_dataset,test_dataset,n_labels

#generate more data with standard augmentation
def gen_eda(text, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):

    # writer = open(output_file, 'w')
    # lines = open(train_orig, 'r').readlines()
    aug_data = []
    # for i, line in enumerate(train):
        # parts = line[:-1].split(',')
    label = text[0]
    sentence = text[1]
    aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
    # for aug_sentence in aug_sentences:
    #     writer.write(label + "\t" + aug_sentence + '\n')
    for aug_sentence in aug_sentences:
        aug_data.append([label,aug_sentence])
    return  aug_data

class loader_data(Dataset):
    def __init__(self,data_text,tokenizer,max_seq_len):
        self.text = data_text[:,1]
        self.labels = data_text[:,0]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labels)

    def get_tokenized(self,text):
        tokens = self.tokenizer.tokenize (text)

        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        text_encode = self.tokenizer.convert_tokens_to_ids (tokens)

        padding = [0] * (self.max_seq_len - len(text_encode))

        text_encode += padding
        return text_encode

    def __getitem__(self, idx):
        text = self.text[idx]
        text_encode = self.get_tokenized(text)
        return (torch.tensor(text_encode),torch.tensor(self.labels[idx]))
