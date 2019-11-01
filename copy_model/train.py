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
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,\
                             weight_decay=0)
GRAD_MAXNORM = 100
BATCH_SIZE = 8 

for epoch in range(NUM_EPOCHS):
    print('Epoch #%d'%epoch)
    bno = 0
    data_gen = get_batches(data)
    while(1):
        count = 0
        losses = []
        for i in range(BATCH_SIZE):
            try:
                q, cxt, cxt_labels, q_labels, mask  = data_gen.__next__()
                prediction = model(q,cxt,cxt_labels, mask).view(-1, num_classes+1)
                l = loss(prediction, q_labels.view(-1))  
                losses.append(l)
                count += 1
            except StopIteration:
                break
        if count < BATCH_SIZE:
            break
        mean_loss = torch.mean(torch.stack(losses))
        optimizer.zero_grad()
        mean_loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), GRAD_MAXNORM)
        optimizer.step()
        if bno %1 == 0:
            print('Loss after batch #%d = %f'%(bno, mean_loss.data))
        bno+=1

