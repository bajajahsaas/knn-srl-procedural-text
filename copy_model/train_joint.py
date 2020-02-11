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
from biobert import *
from bert_crf import *
torch.set_default_tensor_type(torch.DoubleTensor)


# load trained NER system
tokenizer,bert_model = getscibertmodel()

bert_model.cuda()
tagger = BERT_CRF(2*19+1).cuda()
tagger.load_state_dict(torch.load('bert_crf.pt'))
with open('entity_types.txt', 'r') as f:
    etypes = f.read().splitlines()
    edic = {e:i for i, e in enumerate(etypes)}

with open('tagger_dict.pkl', 'rb') as f:
    tagger_data = pickle.load(f)

with open('relations.txt', 'r') as f:
    relations = f.read().splitlines() + ['No relation']
    rel_dic = {r:i for i, r in enumerate(relations)}

with open('buckets.pkl', 'rb') as f:
    buckets = pickle.load(f)

def get_sent_labels(sent, entities):
    tokenized = tokenizer.tokenize(sent)
    enc = tokenizer.encode(sent, add_special_tokens = True)
    labels = ['O' for i in enc]
    for ent, typ, _, span in entities:
        for i in range(span[0]+1, span[1]+1):
            labels[i] = 'I_' + typ
        labels[span[0]] = 'B_' + typ
    return tokenized,enc,labels

def get_entities(sent):
    enc = tokenizer.encode(sent, add_special_tokens = True)
    with torch.no_grad():
        bert_embedding = \
                bert_model(torch.tensor([enc]).cuda())[0].squeeze(0).cpu().numpy()
        pred = tagger(torch.tensor([enc]).cuda())
    entities = []
    i = 0
    start = -1
    while(i<len(enc)):
        # O = outside in BIO
        # print(len(enc))
        # print(len(pred[0]))
        while(i<len(enc) and pred[0][i] == tagger_data['dict']['O']):
            i+=1
        if i >= len(enc):
            break
        tag = tagger_data['list'][pred[0][i]].split('_')[1]
        inside_tag = tagger_data['dict']['I_'+tag]
        start = i
        i += 1
        while(i<len(enc) and pred[0][i] == inside_tag):
            i += 1
        end = i-1
        emb = np.mean(bert_embedding[start:end+1], axis=0)
        entities.append((start, end,emb, edic[tag]))
    return entities

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
        q_sent, ch_text, ct_text,qpos,cpos = datum['query_sent'], \
                datum['context_head_text'], datum['context_tail_text'],\
                datum['query_posdiff'], datum['context_posdiff']
        ch, ct, cl = datum['context_head'], \
                        datum['context_tail'], datum['context_labels']
        cht, ctt =  datum['context_head_type'], datum['context_tail_type']
        
        # create qh, qt using joint prediction
        entities = get_entities(q_sent)
        # compute ((head_start, head_end),(tail_start, tail_end),relation)
        gold_rels = {}
        for h, t, rel in datum['relations']:
            gold_rels[(datum['entities'][h][3],\
                              datum['entities'][t][3])]=rel
        gold_entities = [x[3] for x in datum['entities']]

        qh_pred = []
        qht_pred = []
        qt_pred = []
        qtt_pred = []
        qpos_pred = []
        ql_pred = []
        found_pairs = [] 
        if len(entities) <= 1:
            continue
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i==j:
                    continue
                found_pairs.append(((entities[i][0], entities[i][1]),\
                                    (entities[j][0], entities[j][1])))
                qh_pred.append(entities[i][2])
                qht_pred.append(entities[i][3])
                qt_pred.append(entities[j][2])
                qtt_pred.append(entities[j][3])
                qpos_pred.append(buckets.get_bucket(abs(entities[j][0] -
                                                    entities[i][0])))
                ql_pred.append(gold_rels.get(((entities[i][0],entities[i][1]),(entities[j][0],entities[j][1])),rel_dic['No relation']))
        found_pairs = set(found_pairs)
        labels_not_found = [gold_rels[x]  for x in gold_rels if x not in found_pairs]
        # ql_pred = [rel_dic[x] for x in ql_pred]
        # labels_not_found = [rel_dic[x] for x in labels_not_found]
        
        qh_pred = torch.from_numpy(np.array(qh_pred)).double().unsqueeze(0)
        qht_pred = torch.from_numpy(np.array(qht_pred)).long().unsqueeze(0)
        qt_pred = torch.from_numpy(np.array(qt_pred)).double().unsqueeze(0)
        qtt_pred = torch.from_numpy(np.array(qtt_pred)).long().unsqueeze(0)
        ql_pred = torch.from_numpy(np.array(ql_pred)).long().unsqueeze(0)
        qpos_pred = torch.from_numpy(np.array(qpos_pred)).long().unsqueeze(0)

        if torch.isnan(torch.stack([qh_pred,qt_pred])).any():
            continue

        elif type(cl) != np.ndarray:
            ch,ct,cl,cht,ctt,mask = None, None, None, None, None, None
        else:
            ch = torch.from_numpy(ch).double().unsqueeze(0)
            cl = torch.from_numpy(cl).long().unsqueeze(0)
            ct = torch.from_numpy(ct).double().unsqueeze(0)
            ctt = torch.from_numpy(ctt).long().unsqueeze(0)
            cht = torch.from_numpy(cht).long().unsqueeze(0)
            cpos = torch.from_numpy(cpos).long().unsqueeze(0)
            if torch.isnan(torch.stack([ch,ct])).any():
                continue
            mask = torch.from_numpy(np.ones_like(cl))
        if args.gpu:
            qh_pred = qh_pred.cuda()
            qt_pred = qt_pred.cuda()
            ql_pred = ql_pred.cuda()
            qtt_pred = qtt_pred.cuda()
            qht_pred = qht_pred.cuda()
            qpos_pred = qpos_pred.cuda()
            if ch is not None:
                ch = ch.cuda()
                cl = cl.cuda()
                ct = ct.cuda()
                cht = cht.cuda()
                ctt = ctt.cuda()
                mask = mask.cuda()
                cpos = cpos.cuda()

        yield ((qh_pred, qht_pred),(qt_pred,qtt_pred),qpos_pred),\
                ((ch, cht), (ct, ctt), cpos), cl, ql_pred, mask
        # datum = data[x]
        # qh, qt, ql, ch, ct, cl,qpos,cpos = datum['query_head'], datum['query_tail'],\
        #                 datum['query_labels'], datum['context_head'], \
        #                 datum['context_tail'], datum['context_labels'],\
        #                 datum['query_posdiff'], datum['context_posdiff']
        # qht, qtt, cht, ctt = datum['query_head_type'], datum['query_tail_type'],\
        #         datum['context_head_type'], datum['context_tail_type']
        # qh = torch.from_numpy(qh).unsqueeze(0)
        # qt = torch.from_numpy(qt).unsqueeze(0)
        # ql = torch.from_numpy(ql).unsqueeze(0)
        # qht = torch.from_numpy(qht).unsqueeze(0)
        # qtt = torch.from_numpy(qtt).unsqueeze(0)
        # qpos = torch.from_numpy(qpos).unsqueeze(0)
        # if torch.isnan(torch.stack([qh,qt])).any():
        #     continue
        # elif type(cl) != np.ndarray:
        #     ch,ct,cl,cht,ctt,mask,cpos = None, None, None, None, None, None, None
        # else:
        #     ch = torch.from_numpy(ch).unsqueeze(0)
        #     cl = torch.from_numpy(cl).unsqueeze(0)
        #     ct = torch.from_numpy(ct).unsqueeze(0)
        #     ctt = torch.from_numpy(ctt).unsqueeze(0)
        #     cht = torch.from_numpy(cht).unsqueeze(0)
        #     cpos = torch.from_numpy(cpos).unsqueeze(0)
        #     if torch.isnan(torch.stack([ch,ct])).any():
        #         continue
        #     mask = torch.from_numpy(np.ones_like(cl))
        # if args.gpu:
        #     qh = qh.cuda()
        #     qt = qt.cuda()
        #     ql = ql.cuda()
        #     qtt = qtt.cuda()
        #     qht = qht.cuda()
        #     qpos = qpos.cuda()
        #     if ch is not None:
        #         ch = ch.cuda()
        #         cl = cl.cuda()
        #         ct = ct.cuda()
        #         cht = cht.cuda()
        #         ctt = ctt.cuda()
        #         mask = mask.cuda()
        #         cpos = cpos.cuda()

        # yield ((qh, qht), (qt, qtt), qpos), ((ch, cht), (ct, ctt), cpos), cl, ql, mask


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
