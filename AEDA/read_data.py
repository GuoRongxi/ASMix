from pytransformers import *
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
import torch

# given specified filepath, sort the data and get the dataset for dataloader constructor
def get_dataset(filepath, max_seq_len, trainfile, model = 'bert-base-uncased'):
    # BertTokenizer: a subclass of PreTrainedTokenizerBase provided by huggingface transformers
    # BertTokenizer.from_pretrained: instantiate a BertTokenizer from a predefined tokenizer
    tokenzier = BertTokenizer.from_pretrained(model)

    # train_df = pd.read_csv(filepath + '/train_orig_plus_augs_1.csv', header=None)
    train_df = pd.read_csv(filepath + trainfile, header=None)
    train = np.array(train_df).tolist()
    random.shuffle(train)
    # feed data of ndarray type to tokenizer in __get_tokenized
    train = np.array(train)
    test_df = pd.read_csv(filepath + '/processed_test.csv', header = None)
    test = np.array(test_df).tolist()
    random.shuffle(test)
    test = np.array(test)
    train_dataset = dataset_for_bert(train, tokenzier, max_seq_len)
    test_dataset = dataset_for_bert(test, tokenzier, max_seq_len)
    n_labels = len(set(train[:, 0]))
    return train_dataset, test_dataset, n_labels


class dataset_for_bert(Dataset):
    # data_text : like 'text label'
    def __init__(self, data_text, tokenizer, max_seq_len):
        self.text = data_text[:, 1]
        self.labels = data_text[:, 0]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labels)

    # return an id-list of standard length
    def __get_tokenized(self, text):
        # convert a sentence to tokens
        # Convert a string in s sequence of tokens, using the tokenizer
        # input: str
        # output: List[str]
        tokens = self.tokenizer.tokenize(text)
        # cut off longer sentences than standard
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        # convert tokens to ids
        # input: str or List[str]
        # otput: int or List[int]
        text_encoder = self.tokenizer.convert_tokens_to_ids(tokens)
        # fill up shorter sentences than standard
        padding = [0] * (self.max_seq_len - len(text_encoder))
        text_encoder += padding
        return text_encoder

    def __getitem__(self, idx):
        text = self.text[idx]
        text_encode = self.__get_tokenized(text)
        label = int(self.labels[idx])
        return (torch.tensor(text_encode), torch.tensor(label))




