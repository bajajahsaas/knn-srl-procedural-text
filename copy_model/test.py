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
import sys

nocopy=False
if sys.argv[1] == 'nocopy':
    nocopy = True
MODEL_PATH = sys.argv[2]

EMBEDDING_SIZE = 768
NUM_EPOCHS = 50
num_classes = 31
num_labels = 31

with open('val.pkl', 'rb') as f:
    valdata = pickle.load(f)


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
        if torch.isnan(torch.stack([qh,qt])).any():
            continue
        if nocopy:
            ch=cl=ct=mask=None
        elif type(cl) != np.ndarray:
            mask = None
        else:
            ch = torch.from_numpy(ch).unsqueeze(0)
            cl = torch.from_numpy(cl).unsqueeze(0)
            ct = torch.from_numpy(ct).unsqueeze(0)
            if torch.isnan(torch.stack([ch,ct])).any():
                continue
            mask = torch.from_numpy(np.ones_like(cl))
        yield (qh, qt), (ch, ct), cl, ql, mask


def accuracy(data, model):
    with torch.no_grad():
        all_pred = []
        all_target = []
        for q, cxt, cxt_labels, q_labels, mask  in get_batches(data):
            pred = torch.argmax(model(q, cxt, cxt_labels, mask), \
                                dim=-1).view(-1)
            all_target.append(q_labels.view(-1).data.detach().numpy().copy())
            all_pred.append(pred.data.detach().numpy().copy())
        
        all_pred = np.concatenate(all_pred, 0)
        all_target = np.concatenate(all_target, 0)
        
        for i in range(num_classes+1):
            print('%d %d %d'%(i, np.sum(np.equal(all_target, i)),\
                              np.sum(np.equal(all_pred, i))))
        
        existing_relations = np.not_equal(all_target, num_labels)
        total_accuracy = np.mean(np.equal(all_pred, all_target))
        num_rel_correct = np.sum(existing_relations * np.equal(all_target,\
                                    all_pred))
        accuracy_existing_relations = num_rel_correct/np.sum(existing_relations) 
        return total_accuracy, accuracy_existing_relations


weights = torch.ones(num_classes+1)
weights[-1] = 1.0/20
loss = nn.NLLLoss(weights)
model = CopyEditor(EMBEDDING_SIZE, num_classes)
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,\
                             weight_decay=1e-4)

model = CopyEditor(EMBEDDING_SIZE, num_classes)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


acc1, acc2 = accuracy(valdata, model)
print('Accuracy on val set = %f, Accuracy excluding norel=%f'%(acc1, acc2))