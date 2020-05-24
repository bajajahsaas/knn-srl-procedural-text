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

with open(args.traindata, 'rb') as f:
    traindata = pickle.load(f)


def get_batches(data):
    # only batch size 1 for now
    perm = np.random.permutation(len(data))
    for x in perm:
        datum = data[x]
        qh, qt, ql, ch, ct, cl, qpos, cpos = datum['query_head'], datum['query_tail'], \
                                             datum['query_labels'], datum['context_head'], \
                                             datum['context_tail'], datum['context_labels'], \
                                             datum['query_posdiff'], datum['context_posdiff']
        qht, qtt, cht, ctt = datum['query_head_type'], datum['query_tail_type'], \
                             datum['context_head_type'], datum['context_tail_type']
        qh = torch.from_numpy(qh).unsqueeze(0)
        qt = torch.from_numpy(qt).unsqueeze(0)
        ql = torch.from_numpy(ql).unsqueeze(0)
        qht = torch.from_numpy(qht).unsqueeze(0)
        qtt = torch.from_numpy(qtt).unsqueeze(0)
        qpos = torch.from_numpy(qpos).unsqueeze(0)
        if torch.isnan(torch.stack([qh, qt])).any():
            continue
        elif type(cl) != np.ndarray:
            ch, ct, cl, cht, ctt, mask, cpos = None, None, None, None, None, None, None
        else:
            ch = torch.from_numpy(ch).unsqueeze(0)
            cl = torch.from_numpy(cl).unsqueeze(0)
            ct = torch.from_numpy(ct).unsqueeze(0)
            ctt = torch.from_numpy(ctt).unsqueeze(0)
            cht = torch.from_numpy(cht).unsqueeze(0)
            cpos = torch.from_numpy(cpos).unsqueeze(0)
            if torch.isnan(torch.stack([ch, ct])).any():
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

        # Wet Labs data
        yield ((qh, qht), (qt, qtt), qpos), ((ch, cht), (ct, ctt), cpos), cl, ql, mask

        # Mat Sci data
        # MAX = 500  # MAX queries in one batch
        # for i in range(0, qh.shape[1], MAX):
        #     yield ((qh[:, i:i + MAX, :], qht[:, i:i + MAX]), (qt[:, i:i + MAX, :], \
        #                                                       qtt[:, i:i + MAX]), qpos[:, i:i + MAX]), \
        #           ((ch, cht), (ct, ctt), cpos), cl, ql[:, i:i + MAX], mask


def build_prototype(data, model):
    model.eval()

    protoDict = [None] * (num_classes + 1)
    with torch.no_grad():
        for q, cxt, cxt_labels, q_labels, mask in get_batches(data):
            # q_labels vary from [0, num_classes] (including No_Rel)
            # q_labels is 1-d tensor shape (1, edges)
            query_embedding = model.rel_embedding(q)  # (Batch x edges x dim): (1, n, 256)
            _, edges, dims = query_embedding.shape
            q_labels = q_labels.squeeze(0).cpu().numpy()

            query_embedding = query_embedding.squeeze(0)  # (edges x dim)
            for id, l in enumerate(q_labels):
                if protoDict[l] is None:
                    protoDict[l] = [query_embedding[id]]
                else:
                    protoDict[l].append(query_embedding[id])

    prototypes = {}
    for k in range(num_classes + 1):
        print(len(protoDict[k]), len(protoDict[k][0]))
        result = torch.stack(protoDict[k])  # (edges vs dim)
        prototypes[k] = torch.mean(result, dim=0)

    for k in prototypes.keys():
        print(k, len(prototypes[k]))

    with open(args.test_output_path + "/init_protoypes.pkl", 'wb') as f:
        pickle.dump(prototypes, f)


model = CopyEditor(EMBEDDING_SIZE, args)
if args.gpu:
    model = model.cuda()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

build_prototype(traindata, model)