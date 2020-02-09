import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch import autograd
from torch.utils.data import TensorDataset
from CopyEditor import CopyEditor
import pickle
import torch.optim as optim
import sys
from argparser import args
import matplotlib.pyplot as plt
import os
from EarlyStopping import EarlyStopping

copy = args.copy
generate = args.generate
PLOT_PATH = args.plot_path

EMBEDDING_SIZE = 768
NUM_EPOCHS = args.epochs
MODEL_PATH = args.model_path
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


def get_batches(data):
    # only batch size 1 for now
    perm = np.random.permutation(len(data))
    for x in perm:
        datum = data[x]
        qh, qt, ql, ch, ct, cl,qpos,cpos = datum['query_head'], datum['query_tail'],\
                        datum['query_labels'], datum['context_head'], \
                        datum['context_tail'], datum['context_labels'],\
                        datum['query_posdiff'], datum['context_posdiff']
        qht, qtt, cht, ctt = datum['query_head_type'], datum['query_tail_type'],\
                datum['context_head_type'], datum['context_tail_type']
        qh = torch.from_numpy(qh).unsqueeze(0)
        qt = torch.from_numpy(qt).unsqueeze(0)
        ql = torch.from_numpy(ql).unsqueeze(0)
        qht = torch.from_numpy(qht).unsqueeze(0)
        qtt = torch.from_numpy(qtt).unsqueeze(0)
        qpos = torch.from_numpy(qpos).unsqueeze(0)
        if torch.isnan(torch.stack([qh,qt])).any():
            continue
        elif type(cl) != np.ndarray:
            ch,ct,cl,cht,ctt,mask,cpos = None, None, None, None, None, None, None
        else:
            ch = torch.from_numpy(ch).unsqueeze(0)
            cl = torch.from_numpy(cl).unsqueeze(0)
            ct = torch.from_numpy(ct).unsqueeze(0)
            ctt = torch.from_numpy(ctt).unsqueeze(0)
            cht = torch.from_numpy(cht).unsqueeze(0)
            cpos = torch.from_numpy(cpos).unsqueeze(0)
            if torch.isnan(torch.stack([ch,ct])).any():
                continue
            mask = torch.from_numpy(np.ones_like(cl))
        if args.gpu:
            qh = qh.cuda()
            qt = qt.cuda()
            ql = ql.cuda()
            qtt = qtt.cuda()
            qht = qht.cuda()
            qpos = qpos.cuda()
            if ch is not None:
                ch = ch.cuda()
                cl = cl.cuda()
                ct = ct.cuda()
                cht = cht.cuda()
                ctt = ctt.cuda()
                mask = mask.cuda()
                cpos = cpos.cuda()

        yield ((qh, qht), (qt, qtt), qpos), ((ch, cht), (ct, ctt), cpos), cl, ql, mask


def accuracy(data, model, loss):
    model.eval()
    with torch.no_grad():
        all_pred = []
        all_target = []
        losses = []
        for q, cxt, cxt_labels, q_labels, mask in get_batches(data):
            model_pred = model(q, cxt, cxt_labels, mask)
            pred = torch.argmax(model_pred, dim=-1).view(-1)
            all_target.append(q_labels.view(-1).data.detach().cpu().numpy().copy())
            all_pred.append(pred.data.detach().cpu().numpy().copy())

            log_prob = model_pred.view(-1, num_classes + 1)
            valloss = loss(log_prob, q_labels.view(-1))
            losses.append(valloss)

        all_pred = np.concatenate(all_pred, 0)
        all_target = np.concatenate(all_target, 0)
        total_valloss = torch.sum(torch.stack(losses))

        for i in range(num_classes + 1):
            print('%d %d %d' % (i, np.sum(np.equal(all_target, i)), \
                                np.sum(np.equal(all_pred, i))))

        existing_relations = np.not_equal(all_target, num_labels)
        total_accuracy = np.mean(np.equal(all_pred, all_target))
        num_rel_correct = np.sum(existing_relations * np.equal(all_target, \
                                                               all_pred))
        accuracy_existing_relations = num_rel_correct / np.sum(existing_relations)
        labels = [x for x in range(num_labels)]  # All possible output labels for multiclass problem
        f1 = f1_score(all_target, all_pred, labels=labels, average="micro")
        return total_accuracy, accuracy_existing_relations, total_valloss, f1


def gridSearchDownSample():
    weights_tune = [1, 5, 10, 15, 20]
    loss_to_plot = {}
    val_loss_to_plot = {}
    f1_to_plot = {}

    bestf1 = 0
    bestweight = 0

    for downsample in weights_tune:
        this_model_path = MODEL_PATH[:len(MODEL_PATH) - 3] + str(downsample) + MODEL_PATH[len(MODEL_PATH) - 3:]
        print("Training with Downsampling param as ", downsample)
        weights = torch.ones(num_classes + 1)
        weights[-1] = 1.0 / downsample
        if args.gpu:
            weights = weights.cuda()
        loss = nn.NLLLoss(weights)
        model = CopyEditor(EMBEDDING_SIZE, args)
        if args.gpu:
            model = model.cuda()
        learning_rate = args.lr
        weight_decay = args.weight_decay
        print('parameters:')
        print(model.parameters)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, \
                                     weight_decay=weight_decay)
        early_stopping = EarlyStopping(patience=50, model_path=this_model_path, minmax= \
            'max')

        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0.0
            print('Epoch #%d' % epoch)
            model.train()
            bno = 0
            data_gen = get_batches(traindata)
            while (1):
                count = 0
                losses = []
                for i in range(BATCH_SIZE):
                    try:
                        q, cxt, cxt_labels, q_labels, mask = data_gen.__next__()
                        prediction = model(q, cxt, cxt_labels, mask).view(-1, num_classes + 1)
                        l = loss(prediction, q_labels.view(-1))
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
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_MAXNORM)
                    optimizer.step()
                except:
                    print('No grad batch')
                    pass

                if bno % 100 == 0:
                    print('Loss after batch #%d = %f' % (bno, total_loss.data))

                epoch_loss += total_loss.data
                bno += 1

            if downsample not in loss_to_plot:  # first epoch for hyperparam
                loss_to_plot[downsample] = []
                val_loss_to_plot[downsample] = []
                f1_to_plot[downsample] = []

            loss_to_plot[downsample].append(epoch_loss)
            acc1, acc2, valloss, f1score = accuracy(valdata, model, loss)
            print('Accuracy on val set = %f, Accuracy excluding norel=%f' % (acc1, acc2))
            val_loss_to_plot[downsample].append(valloss)
            f1_to_plot[downsample].append(f1score)

            early_stopping.step(f1score, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        _, _, _, f1score = accuracy(valdata, model, loss)
        print("Downsampling with ", downsample, " F1 Score = ", f1score)
        if f1score > bestf1:
            bestf1 = f1score
            bestweight = downsample

        plt.subplot(2, 1, 1)
        plt.plot(loss_to_plot[bestweight], 'b', label='Training Loss')
        plt.plot(val_loss_to_plot[bestweight], 'r', label='Validation Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.subplot(2, 1, 2)
        plt.plot(f1_to_plot[bestweight], 'o', label='Validation F1 score')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')

        plt.savefig(PLOT_PATH + "/loss_plot.png")

    return bestweight, bestf1


def noGridSearch(downsample):
    loss_to_plot = []
    val_loss_to_plot = []
    f1_to_plot = []
    weights = torch.ones(num_classes + 1)
    weights[-1] = 1.0 / downsample
    if args.gpu:
        weights = weights.cuda()
    loss = nn.NLLLoss(weights)
    model = CopyEditor(EMBEDDING_SIZE, args)
    if args.gpu:
        model = model.cuda()
    learning_rate = args.lr
    weight_decay = args.weight_decay
    print('parameters:')
    print(model.parameters)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, \
                                 weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=50, model_path=MODEL_PATH, minmax= \
        'max')

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        print('Epoch #%d' % epoch)
        model.train()
        bno = 0
        data_gen = get_batches(traindata)
        while (1):
            count = 0
            losses = []
            for i in range(BATCH_SIZE):
                try:
                    q, cxt, cxt_labels, q_labels, mask = data_gen.__next__()
                    prediction = model(q, cxt, cxt_labels, mask).view(-1, num_classes + 1)
                    l = loss(prediction, q_labels.view(-1))
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
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_MAXNORM)
                optimizer.step()
            except:
                print('No grad batch')
                pass

            if bno % 100 == 0:
                print('Loss after batch #%d = %f' % (bno, total_loss.data))

            epoch_loss += total_loss.data
            bno += 1

        loss_to_plot.append(epoch_loss)
        acc1, acc2, valloss, f1score = accuracy(valdata, model, loss)
        print('Accuracy on val set = %f, Accuracy excluding norel=%f' % (acc1, acc2))
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


downsample_arg = args.downsample

if downsample_arg == 0:
    bestweight, bestf1 = gridSearchDownSample()
    print("Best downsampling parameter = ", bestweight, " with F1 Score = ", bestf1)
else:
    noGridSearch(downsample_arg)
