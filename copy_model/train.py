import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch import autograd
from torch.utils.data import TensorDataset
from CopyEditor import CopyEditorBertWrapper
import pickle
import torch.optim as optim
import sys
from argparser import args
import matplotlib.pyplot as plt
import os
from EarlyStopping import EarlyStopping

copy = args.copy
generate = args.generate
MODEL_PATH = args.model_path
PLOT_PATH = args.plot_path

EMBEDDING_SIZE = 768
NUM_EPOCHS = args.epochs
num_classes = args.classes
num_labels = args.classes
BATCH_SIZE = args.batch_size
GRAD_MAXNORM = args.grad_maxnorm

# Creates folder for all logs (incl plots)
if not os.path.exists(PLOT_PATH):
    os.makedirs(PLOT_PATH)

with open(args.traindata, 'rb') as f:
    traindata = pickle.load(f)

with open(args.valdata, 'rb') as f:
    valdata = pickle.load(f)

# weights = torch.ones(num_classes+1)
# weights[-1] = 1.0/10
# if args.gpu:
#     weights = weights.cuda()
loss = nn.NLLLoss()
model = CopyEditorBertWrapper(EMBEDDING_SIZE, num_classes, copy=args.copy,
                   generate=args.generate)
if args.gpu:
        model = model.cuda()
learning_rate = args.lr
weight_decay = args.weight_decay
print('parameters:')
print(model.parameters)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,\
                             weight_decay=weight_decay)
early_stopping = EarlyStopping(patience=20, model_path=MODEL_PATH,minmax = \
                               'max')

def get_batches(data):
    # only batch size 1 for now
    perm = np.random.permutation(len(data))
    for x in perm:
        datum = data[x]
        qtokens, qrelations, qlabels = datum['query_tokens'], \
            datum['query_relations'], datum['query_labels']
        ctokens, crelations, clabels = datum['context_tokens'], \
            datum['context_relations'], datum['context_labels']
        qtokens = torch.from_numpy(qtokens)
        qrelations = torch.from_numpy(qrelations)
        qlabels = torch.from_numpy(qlabels)
        ctokens = [torch.from_numpy(x) for x in ctokens]
        crelations = [torch.from_numpy(x) for x in crelations]
        clabels = [torch.from_numpy(x) for x in clabels]
        if args.gpu:
            qtokens = qtokens.cuda()
            qrelations = qrelations.cuda()
            qlabels = qlabels.cuda()
            for i in range(len(ctokens)):
                ctokens[i] = ctokens[i].cuda()
                crelations[i] = crelations[i].cuda()
                clabels[i] = clabels[i].cuda()

        yield qtokens, qrelations, qlabels, ctokens, crelations, clabels


def accuracy(data, model):
    model.eval()
    with torch.no_grad():
        all_pred = []
        all_target = []
        losses = []
        for qt, qr,ql,ct, cr, cl  in get_batches(data):
            model_pred, _ = model(qt, qr, ct, cr, cl)
            pred = torch.argmax(model_pred, dim=-1).view(-1)
            all_target.append(ql.view(-1).data.detach().cpu().numpy().copy())
            all_pred.append(pred.data.detach().cpu().numpy().copy())

            log_prob = model_pred.view(-1, num_classes + 1)
            valloss = loss(log_prob, ql.view(-1))
            losses.append(valloss)

        all_pred = np.concatenate(all_pred, 0)
        all_target = np.concatenate(all_target, 0)
        total_valloss = torch.sum(torch.stack(losses))

        for i in range(num_classes+1):
            print('%d %d %d'%(i, np.sum(np.equal(all_target, i)),\
                              np.sum(np.equal(all_pred, i))))

        existing_relations = np.not_equal(all_target, num_labels)
        total_accuracy = np.mean(np.equal(all_pred, all_target))
        num_rel_correct = np.sum(existing_relations * np.equal(all_target,\
                                    all_pred))
        accuracy_existing_relations = num_rel_correct/np.sum(existing_relations)
        labels = [x for x in range(num_labels)]  # All possible output labels for multiclass problem
        f1 = f1_score(all_target, all_pred, labels=labels, average="micro")
        return total_accuracy, accuracy_existing_relations, total_valloss, f1


loss_to_plot = []
val_loss_to_plot = []
f1_to_plot = []
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    print('Epoch #%d'%epoch)
    model.train()
    bno = 0
    data_gen = get_batches(traindata)
    acc1, acc2, valloss, f1score = accuracy(valdata, model)
    print('Accuracy on val set = %f, Accuracy excluding norel=%f'%(acc1, acc2))
    while(1):
        count = 0
        losses = []
        for i in range(BATCH_SIZE):
            try:
                qt, qr, ql, ct, cr, cl  = data_gen.__next__()
                prediction,_ = model(qt, qr, ct, cr, cl)
                l = loss(prediction.view(-1, num_classes+1), ql.view(-1))
                print(l)
                losses.append(l)
                count += 1
            except StopIteration:
                break
        if count < BATCH_SIZE:
            break
        total_loss = torch.sum(torch.stack(losses))
        optimizer.zero_grad()

        # This is required for when --no-generate is set and No context is
        # avalable. The is no gradient function for the loss in this case.
        try:
            total_loss.backward()
            print(model.copy_editor.should_copy.network.weight.grad)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_MAXNORM)
            optimizer.step()
        except KeyboardInterrupt:
            print('No grad batch')
            pass

        if bno %100 == 0:
            print('Loss after batch #%d = %f'%(bno, total_loss.data))

        epoch_loss += total_loss.data
        bno+=1

    loss_to_plot.append(epoch_loss)
    acc1, acc2, valloss, f1score = accuracy(valdata, model)
    print('Accuracy on val set = %f, Accuracy excluding norel=%f'%(acc1, acc2))
    val_loss_to_plot.append(valloss)
    f1_to_plot.append(f1score)

    early_stopping.step(f1score, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break


plt.subplot(2, 1, 1)
plt.plot(loss_to_plot, 'b', label='Training Loss')
plt.plot(val_loss_to_plot, 'r', label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(f1_to_plot, 'o', label='Validation F1 score')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('F1 Score')

plt.savefig(PLOT_PATH + "/loss_plot.png")

