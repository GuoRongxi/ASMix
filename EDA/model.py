import torch.nn as nn
from pytransformers import *
import torch

class VanillaBert(nn.Module):
    def __init__(self, num_labels = 2, model = 'bert-base-uncased'):
        super().__init__()
        # instantiate a BertModel from a pre-trained model configuration
        self.bert = BertModel.from_pretrained(model)
        self.MLP = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))

    def forward(self, input):
        # bert forward expect input as
        # torch.LongTensor or shape (batch_size, sequence_length)
        # tensor means tensor(List[ids: int])
        # return: last_hidden_states
        # (torch.FloatTensor) of shape (batch_size, sequence_length, hidden_size)
        all_hidden, _ = self.bert(input)
        # (batch_size, sequence_length, hidden_size) means to (batch_size, hidden_size)
        # calculate mean value of specified hidden states of every token
        # get the hidden representation of a whole sentence, equal to combine all represents of tokens in a sentence
        # shape of (batch_size, hidden_size)
        pooled_output = torch.mean(all_hidden, 1)
        # through a hidden layer and an output layer
        # indicate hidden_size is 768
        predict = self.MLP(pooled_output)
        return predict
