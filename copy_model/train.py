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
from argparser import args


copy = args.copy
generate = args.generate
MODEL_PATH = args.model_path

EMBEDDING_SIZE = 768
NUM_EPOCHS = args.epochs
num_classes = args.classes
num_labels = args.classes
BATCH_SIZE = args.batch_size
GRAD_MAXNORM = args.grad_maxnorm

with open(args.traindata, 'rb') as f:
    traindata = pickle.load(f)

with open(args.valdata, 'rb') as f:
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
        elif type(cl) != np.ndarray:
            ch,ct,cl,mask = None, None, None, None
        else:
            ch = torch.from_numpy(ch).unsqueeze(0)
            cl = torch.from_numpy(cl).unsqueeze(0)
            ct = torch.from_numpy(ct).unsqueeze(0)
            if torch.isnan(torch.stack([ch,ct])).any():
                continue
            mask = torch.from_numpy(np.ones_like(cl))
        if args.gpu:
            qh = qh.cuda()
            qt = qt.cuda()
            ql = ql.cuda()
            if ch is not None:
                ch = ch.cuda()
                cl = cl.cuda()
                ct = ct.cuda()
                mask = mask.cuda()

        yield (qh, qt), (ch, ct), cl, ql, mask


def accuracy(data, model):
    with torch.no_grad():
        all_pred = []
        all_target = []
        for q, cxt, cxt_labels, q_labels, mask  in get_batches(data):
            pred = torch.argmax(model(q, cxt, cxt_labels, mask), \
                                dim=-1).view(-1)
            all_target.append(q_labels.view(-1).data.detach().cpu().numpy().copy())
            all_pred.append(pred.data.detach().cpu().numpy().copy())
        
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
if args.gpu:
    weights = weights.cuda()
loss = nn.NLLLoss(weights)
model = CopyEditor(EMBEDDING_SIZE, num_classes, copy=args.copy,
                   generate=args.generate)
if args.gpu:
        model = model.cuda()
learning_rate = args.lr
weight_decay = args.weight_decay
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,\
                             weight_decay=weight_decay)



for epoch in range(NUM_EPOCHS):
    print('Epoch #%d'%epoch)
    acc1, acc2 = accuracy(valdata, model)
    print('Accuracy on val set = %f, Accuracy excluding norel=%f'%(acc1, acc2))
    bno = 0
    data_gen = get_batches(traindata)
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

        # This is required for when --no-generate is set and No context is
        # avalable. The is no gradient function for the loss in this case.
        try:
            mean_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_MAXNORM)
            optimizer.step()
        except:
            print('No grad batch')
            pass

        if bno %100 == 0:
            print('Loss after batch #%d = %f'%(bno, mean_loss.data))
        bno+=1

torch.save(model.state_dict(), MODEL_PATH)
