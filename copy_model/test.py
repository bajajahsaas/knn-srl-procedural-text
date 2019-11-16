import numpy as np
import torch
from CopyEditor import CopyEditor
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

with open(args.valdata, 'rb') as f:
    valdata = pickle.load(f)

relations = ['Measure-Type-Link',
             'Of-Type',
             'Coreference-Link',
             'Count',
             'Or',
             'Using',
             'Acts-on',
             'Meronym',
             'Site',
             'Setting',
             'Mod-Link',
             'Measure',
             'Creates',
             'Misc-Link',
             'No-Rel']


def get_batches(data):
    # only batch size 1 for now
    perm = np.random.permutation(len(data))
    for x in perm:
        datum = data[x]
        qh, qt, ql, ch, ct, cl = datum['query_head'], datum['query_tail'], \
                                 datum['query_labels'], datum['context_head'], \
                                 datum['context_tail'], datum['context_labels']
        qh = torch.from_numpy(qh).unsqueeze(0)
        qt = torch.from_numpy(qt).unsqueeze(0)
        ql = torch.from_numpy(ql).unsqueeze(0)
        if torch.isnan(torch.stack([qh, qt])).any():
            continue
        elif type(cl) != np.ndarray:
            ch, ct, cl, mask = None, None, None, None
        else:
            ch = torch.from_numpy(ch).unsqueeze(0)
            cl = torch.from_numpy(cl).unsqueeze(0)
            ct = torch.from_numpy(ct).unsqueeze(0)
            if torch.isnan(torch.stack([ch, ct])).any():
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
        labels = [x for x in range(num_labels)]  # All possible output labels for multiclass problem

        all_pred = []
        all_target = []
        precision_sentences = []
        recall_sentences = []
        f1_sentences = []

        for q, cxt, cxt_labels, q_labels, mask in get_batches(data):
            pred = torch.argmax(model(q, cxt, cxt_labels, mask), \
                                dim=-1).view(-1)

            this_target = q_labels.view(-1).data.detach().numpy().copy()
            this_pred = pred.data.detach().numpy().copy()

            precision_sentences.append(precision_score(this_target, this_pred, labels=labels, average="micro"))
            recall_sentences.append(recall_score(this_target, this_pred, labels=labels, average="micro"))
            f1_sentences.append(f1_score(this_target, this_pred, labels=labels, average="micro"))

            all_target.append(this_target)
            all_pred.append(this_pred)

        all_pred = np.concatenate(all_pred, 0)
        all_target = np.concatenate(all_target, 0)
        # both of these have lengths = num_labels + 1
        print('Prediction list size = ', len(list(set(all_pred))))
        print('Target list size = ', len(list(set(all_target))))

        print('Micro precision is ', str(round(precision_score(all_target, all_pred, labels=labels, average="micro"), 2)))
        print('Macro precision is ', str(round(precision_score(all_target, all_pred, labels=labels, average="macro"), 2)))

        precisionlist = precision_score(all_target, all_pred, labels=labels, average=None)
        # print('Macro precision excluding no rel', np.mean(precisionlist[:len(precisionlist) - 1]))

        # print('Per class precision is')
        # for rel, precision in zip(relations, precisionlist):
        #     print(rel, precision)

        print('Micro recall is ', str(round(recall_score(all_target, all_pred, labels=labels, average="micro"),2)))
        print('Macro recall is ', str(round(recall_score(all_target, all_pred, labels=labels, average="macro"),2)))

        recalllist = recall_score(all_target, all_pred, labels=labels, average=None)
        # print('Macro recall excluding no rel', np.mean(recalllist[:len(recalllist) - 1]))

        # print('Per class recall is')
        # for rel, recall in zip(relations, recalllist):
        #     print(rel, recall)

        print('Micro F1 score is ', str(round(f1_score(all_target, all_pred, labels=labels, average="micro"), 2)))
        print('Macro F1 score is ', str(round(f1_score(all_target, all_pred, labels=labels, average="macro"), 2)))

        f1list = f1_score(all_target, all_pred, labels=labels, average=None)
        # print('Macro f1 excluding no rel', np.mean(f1list[:len(f1list) - 1]))

        # print('Per class F1 score is')
        # for rel, f1 in zip(relations, f1list):
        #     print(rel, f1)

        # for i in range(num_classes+1):
        #     print('%d %d %d'%(i, np.sum(np.equal(all_target, i)),\
        #                       np.sum(np.equal(all_pred, i))))

        # value = num_label means "no-rel"
        existing_relations = np.not_equal(all_target, num_labels)
        total_accuracy = np.mean(np.equal(all_pred, all_target))
        num_rel_correct = np.sum(existing_relations * np.equal(all_target, all_pred))
        accuracy_existing_relations = num_rel_correct / np.sum(existing_relations)

        print('Macro Average Precision on Sentences ', str(round(np.mean(precision_sentences),2)))
        print('Macro Average Recall on Sentences ', str(round(np.mean(recall_sentences),2)))
        print('Macro Average F1 on Sentences ', str(round(np.mean(f1_sentences),2)))

        return total_accuracy, accuracy_existing_relations


model = CopyEditor(EMBEDDING_SIZE, num_classes, copy=args.copy,
                   generate=args.generate)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

acc1, acc2 = accuracy(valdata, model)
print('Accuracy on val set = ', str(round(acc1, 2)), 'Accuracy excluding norel ', str(round(acc2,2)))
