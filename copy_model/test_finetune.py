import os

import numpy as np
import csv
import torch
from CopyEditor import CopyEditorBertWrapper
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
from argparser import args

copy = args.copy
generate = args.generate
MODEL_PATH = args.model_path

EMBEDDING_SIZE = 768
NUM_EPOCHS = args.epochs
num_classes = args.classes
num_labels = args.classes
MAX_QUERY = 100

with open(args.valdata, 'rb') as f:
    valdata = pickle.load(f)

# relations = ['Measure-Type-Link',
#              'Of-Type',
#              'Coreference-Link',
#              'Count',
#              'Or',
#              'Using',
#              'Acts-on',
#              'Meronym',
#              'Site',
#              'Setting',
#              'Mod-Link',
#              'Measure',
#              'Creates',
#              'Misc-Link',
#              'No-Rel']

relations = open('relations.txt', 'r').read().splitlines() + ['No-Rel']


def get_batches(data):
    # only batch size 1 for now
    perm = np.random.permutation(len(data))
    for x in perm:
        datum = data[x]
        qtokens, qrelations, qlabels = datum['query_tokens'], \
            datum['query_relations'], datum['query_labels']
        ctokens, crelations, clabels = datum['context_tokens'], \
            datum['context_relations'], datum['context_labels']
        qrelations = torch.from_numpy(qrelations)
        qlabels = torch.from_numpy(qlabels)
        crelations = [torch.from_numpy(x) for x in crelations]
        clabels = [torch.from_numpy(x) for x in clabels]
        if args.gpu:
            qrelations = qrelations.cuda()
            qlabels = qlabels.cuda()
            for i in range(len(ctokens)):
                crelations[i] = crelations[i].cuda()
                clabels[i] = clabels[i].cuda()
        for start in range(0,qrelations.shape[0], MAX_QUERY):
            yield qtokens,\
                qrelations[start:start+MAX_QUERY],\
                qlabels[start:start+MAX_QUERY], ctokens,\
                crelations, clabels

def accuracy(data, model):
    with torch.no_grad():
        labels = [x for x in range(num_labels)]  # All possible output labels for multiclass problem

        all_pred = []
        all_target = []
        precision_sentences = []
        recall_sentences = []
        f1_sentences = []

        f = open(os.path.join(args.test_output_path,'predictions.csv'), 'w')
        writer = csv.writer(f)
        writer.writerow(['Sentence', 'Relations in context', 'Head', 'Tail',
                         'Target', 'Prediction', 'correct', 'copyprob'])

        def write_csv_row(writer, sent, qtext, qlabels, ctext, clabels, pred, copyprob):
            copyprob = torch.exp(copyprob.view(-1))
            qlabels = qlabels.view(-1)
            if clabels is not None:
                clabels = clabels.view(-1)
            pred = pred.view(-1)
            qh,qt = qtext
            ch,ct = ctext
            if clabels is not None:
                # copyable = '\n'.join(['%s \t\t %s \t\t %s'%(a,relations[b], c) for
                #                   (a,b,c) in zip(ch,clabels, ct)])
                copyable = '|'.join(list(set([relations[x] for x in clabels])))
            else:
                copyable = 'No Context found'
            for a,b,c,d,e in zip(qh,qt,qlabels, pred, copyprob):
                writer.writerow([sent, copyable, a, b, relations[c],
                                 relations[d], c==d, '%.5f'%e])





        for q_tokens, q_relations, q_labels,\
                cxt_tokens, cxt_relations, cxt_labels in get_batches(data):
            logprob, copyprob = model(q_tokens, q_relations, \
                                      cxt_tokens, cxt_relations, cxt_labels)
            pred = torch.argmax(logprob, \
                                dim=-1).view(-1)

            this_target = q_labels.view(-1).data.cpu().detach().numpy().copy()
            this_pred = pred.data.cpu().detach().numpy().copy()
            print(this_target)
            print(this_pred)

            precision_sentences.append(precision_score(this_target, this_pred,\
                                                       labels=labels, average="micro"))
            recall_sentences.append(recall_score(this_target, this_pred, labels=labels, average="micro"))
            f1_sentences.append(f1_score(this_target, this_pred, labels=labels, average="micro"))

            all_target.append(this_target)
            all_pred.append(this_pred)
            # write_csv_row(writer,sent, q_text, q_labels, cxt_text, cxt_labels, pred, copyprob)
        f.close()
        all_pred = np.concatenate(all_pred, 0)
        all_target = np.concatenate(all_target, 0)
        # both of these have lengths = num_labels + 1
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

        macro_sent_prec = str(round(np.mean(precision_sentences),4)*100)
        macro_sent_rec = str(round(np.mean(recall_sentences),4)*100)
        macro_sent_f1 = str(round(np.mean(f1_sentences),4)*100)

        writer_sc.writerow(['microaverage', micro_prec, micro_rec, micro_f1])
        writer_sc.writerow(['macroaverage', macro_prec, macro_rec, macro_f1])
        writer_sc.writerow(['microaverage_sent', macro_sent_prec, macro_sent_rec, macro_sent_f1])

        # print('Per class F1 score is')
        writer_sc.writerow(['\t', '\t', '\t', '\t'])
        writer_sc.writerow(['Relation', relations[:len(relations) - 1]]) #No-rel accuracy is not returned
        f1_class = []
        for rel, f1 in zip(relations, f1list):
            f1_class.append(str(round(f1, 4)*100))

        writer_sc.writerow(['F1 score', f1_class])

        return total_accuracy, accuracy_existing_relations


model = CopyEditorBertWrapper(EMBEDDING_SIZE, args)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()
model.cuda()

acc1, acc2 = accuracy(valdata, model)
print('Accuracy on val set = ', str(round(acc1, 4)*100), 'Accuracy excluding norel ', str(round(acc2, 4)*100))
