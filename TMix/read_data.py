import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import *
import torch.utils.data as Data
import pickle


def get_data(data_path, n_labeled_per_class, max_seq_len=150, model='bert-base-uncased', train_aug=False):
    """Read data, split the dataset, and build dataset for dataloaders.

    Arguments:
        data_path {str} -- Path to your dataset folder: contain a train.csv and test.csv
        n_labeled_per_class {int} -- Number of labeled data per class

    Keyword Arguments:
        unlabeled_per_class {int} -- Number of unlabeled data per class (default: {5000})
        max_seq_len {int} -- Maximum sequence length (default: {256})
        model {str} -- Model name (default: {'bert-base-uncased'})
        train_aug {bool} -- Whether performing augmentation on labeled training set (default: {False})

    """
    # Load the tokenizer for bert
    tokenizer = BertTokenizer.from_pretrained(model)

    train_df = pd.read_csv(data_path+'processed_train.csv', header=None)
    test_df = pd.read_csv(data_path+'processed_test.csv', header=None)

    # Here we only use the bodies and removed titles to do the classifications
    train_labels = np.array(train_df[0])
    train_text = np.array(train_df[1])

    test_labels = np.array(test_df[0])
    test_text = np.array(test_df[1])

    n_labels = max(test_labels) + 1

    # Build the dataset class for each set
    train_labeled_dataset = loader_labeled(
        train_text, train_labels, tokenizer, max_seq_len, train_aug)
    test_dataset = loader_labeled(
        test_text, test_labels, tokenizer, max_seq_len)


    return train_labeled_dataset, test_dataset, n_labels


class loader_labeled(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len, aug=False):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.trans_dist = {}


    def __len__(self):
        return len(self.labels)


    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def __getitem__(self, idx):
        text = self.text[idx]
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        return (torch.tensor(encode_result), self.labels[idx], length)

