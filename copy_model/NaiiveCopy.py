import numpy as np
import os
import csv
import torch
from CopyEditor import CopyEditor
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
from argparser import args
import re
from tqdm import tqdm

num_labels = args.classes 
relations = open('relations.txt', 'r').read().splitlines() + ['No-Rel']
labels = [x for x in range(num_labels)]  # All possible output labels for multiclass problem



def filter(st):
    return re.sub("\d+", "_NUM_", st).lower()

def naiive_string_baseline(query_head, query_tail, context_head,\
                           context_tail, context_labels):
    num_classes = num_labels + 1
    predictions = []
    for i in range(len(query_head)):
        labelcount = [0] * (num_classes)
        if len(context_head) >0:
            for j in range(len(context_head)):
                if filter(query_head[i]) == filter(context_head[j]) \
                   and filter(query_tail[i]) == filter(context_tail[j]):
                    labelcount[context_labels[j]] += 1
            predictions.append(np.argmax(labelcount))
        else:
            # no context
            predictions.append(num_labels)
    return np.asarray(predictions)



# copy matching entity types
def naiive_prediction(query_head_type, query_tail_type, context_head_type,\
                      context_tail_type, context_labels):
    num_classes = num_labels + 1
    predictions = []
    for i in range(query_head_type.shape[0]):
        labelcount = [0] * (num_classes)
        if context_head_type is not None:
            for j in range(context_head_type.shape[0]):
                if query_head_type[i] == context_head_type[j] \
                   and query_tail_type[i] == context_tail_type[j]:
                    labelcount[context_labels[j]] += 1
            predictions.append(np.argmax(labelcount))
        else:
            predictions.append(num_labels)
    return np.asarray(predictions)

def evaluate(all_pred, all_target):
        print('Prediction list size = ', len(list(set(all_pred))))
        print('Target list size = ', len(list(set(all_target))))

        f = open(os.path.join(args.test_output_path, 'scores.csv'), 'w')
        writer_sc = csv.writer(f)
        writer_sc.writerow(['Averaging', 'Precision', 'Recall', 'F1 Score'])

        micro_prec =  str(round(precision_score(all_target, all_pred, labels=labels, average="micro"), 4) * 100)
        macro_prec = str(round(precision_score(all_target, all_pred, labels=labels, average="macro"), 4) * 100)

        precisionlist = precision_score(all_target, all_pred, labels=labels, average=None)
        # print('Macro precision excluding no rel', np.mean(precisionlist[:len(precisionlist) - 1]))

        # print('Per class precision is')
        # for rel, precision in zip(relations, precisionlist):
        #     print(rel, precision)

        micro_rec =  str(round(recall_score(all_target, all_pred, labels=labels, average="micro"),4)*100)
        macro_rec = str(round(recall_score(all_target, all_pred, labels=labels, average="macro"),4)*100)

        recalllist = recall_score(all_target, all_pred, labels=labels, average=None)
        # print('Macro recall excluding no rel', np.mean(recalllist[:len(recalllist) - 1]))

        # print('Per class recall is')
        # for rel, recall in zip(relations, recalllist):
        #     print(rel, recall)

        micro_f1 = str(round(f1_score(all_target, all_pred, labels=labels, average="micro"), 4)*100)
        macro_f1 = str(round(f1_score(all_target, all_pred, labels=labels, average="macro"), 4)*100)

        f1list = f1_score(all_target, all_pred, labels=labels, average=None)
        # print('Macro f1 excluding no rel', np.mean(f1list[:len(f1list) - 1]))

        # for i in range(num_classes+1):
        #     print('%d %d %d'%(i, np.sum(np.equal(all_target, i)),\
        #                       np.sum(np.equal(all_pred, i))))

        # value = num_label means "no-rel"
        existing_relations = np.not_equal(all_target, num_labels)
        total_accuracy = np.mean(np.equal(all_pred, all_target))
        num_rel_correct = np.sum(existing_relations * np.equal(all_target, all_pred))
        accuracy_existing_relations = num_rel_correct / np.sum(existing_relations)


        writer_sc.writerow(['microaverage', micro_prec, micro_rec, micro_f1])
        writer_sc.writerow(['macroaverage', macro_prec, macro_rec, macro_f1])

        # print('Per class F1 score is')
        writer_sc.writerow(['\t', '\t', '\t', '\t'])
        writer_sc.writerow(['Relation', relations[:len(relations) - 1]]) #No-rel accuracy is not returned
        f1_class = []
        for rel, f1 in zip(relations, f1list):
            f1_class.append(str(round(f1, 4)*100))

        writer_sc.writerow(['F1 score', f1_class])

        return total_accuracy, accuracy_existing_relations

with open(args.valdata, 'rb') as f:
    valdata = pickle.load(f)


qlabels = []
qpreds = []
for datum in tqdm(valdata):
    # qh, qt, ch, ct, ql, cl = datum['query_head_type'], datum['query_tail_type'],\
    #                     datum['context_head_type'], datum['context_tail_type'],\
    #                     datum['query_labels'], datum['context_labels']
    # qpred = naiive_prediction(qh, qt, ch, ct, cl)
    qh, qt, ch, ct, ql, cl = datum['query_head_text'], datum['query_tail_text'],\
                        datum['context_head_text'], datum['context_tail_text'],\
                        datum['query_labels'], datum['context_labels']
    qpred = naiive_string_baseline(qh, qt, ch, ct, cl)
    qpreds.append(qpred)
    qlabels.append(ql)

all_labels = np.concatenate(qlabels)
all_predictions = np.concatenate(qpreds)

evaluate(all_predictions, all_labels)
