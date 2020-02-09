import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from biobert import *
from torchcrf import CRF


class BERT_CRF(nn.Module):
    def __init__(self, num_labels, dim = 768):
        super(BERT_CRF, self).__init__()
        _, self.bert = getscibertmodel()
        self.emitter = nn.Linear(dim, num_labels) 
        self.crf = CRF(num_labels)

    def forward(self, sentence, labels=None):
        # sentence = 1 x seq_len
        # labels = 1 x seq_len
        with torch.no_grad():
            emb = self.bert(sentence)[0].squeeze(0)

        # seq_length x dim
        emission = self.emitter(emb)
        # seq_length x num_labels
        if labels is not None:
            # return NLL
            return -self.crf(emission.unsqueeze(1),\
                            labels.squeeze(0).unsqueeze(1))
        else:
            return self.crf.decode(emission.unsqueeze(1))
