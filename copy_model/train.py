import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.utils.data import TensorDataset
from CopyEditor import CopyEditor
import pickle
import torch.optim as optim

EMBEDDING_SIZE = 768
NUM_EPOCHS = 10
num_classes = 9
BATCH_SIZE = 16

with open('dataset.pkl', 'rb') as f:
    data = pickle.load(f)


def get_batches(data):
    # only batch size 1 for now
    perm = np.random.permutation(len(data))
    for x in perm:
        datum = data[x]
        qh, qt, ql, ch, ct, cl = datum['query_head'], datum['query_tail'],\
                        datum['query_labels'], datum['context_head'], \
                        datum['context_tail'], datum['context_labels']
        qh = torch.from_numpy(qh).unsqueeze(0)
        qt = torch.from_numpy(qt).unsqueeze(0)
        ql = torch.from_numpy(ql).unsqueeze(0)
        if type(cl) != np.ndarray:
            mask = None
        else:
            ch = torch.from_numpy(ch).unsqueeze(0)
            cl = torch.from_numpy(cl).unsqueeze(0)
            ct = torch.from_numpy(ct).unsqueeze(0)
            mask = torch.from_numpy(np.ones_like(cl))
        yield (qh, qt), (ch, ct), cl, ql, mask

loss = nn.NLLLoss()
model = CopyEditor(EMBEDDING_SIZE, num_classes)
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(NUM_EPOCHS):
    for q, cxt, cxt_labels, q_labels, mask in get_batches(data):
        
        with autograd.detect_anomaly():
            prediction = model(q,cxt,cxt_labels, mask).view(-1, num_classes+1)
            l = loss(torch.log(prediction), q_labels.view(-1))  
            print(l)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        

