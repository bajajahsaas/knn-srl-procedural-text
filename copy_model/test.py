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
        for q, cxt, cxt_labels, q_labels, mask in get_batches(data):
            pred = torch.argmax(model(q, cxt, cxt_labels, mask), \
                                dim=-1).view(-1)

            this_target = q_labels.view(-1).data.detach().numpy().copy()
            this_pred = pred.data.detach().numpy().copy()

            all_target.append(this_target)
            all_pred.append(this_pred)


        all_pred = np.concatenate(all_pred, 0)
        all_target = np.concatenate(all_target, 0)
        # both of these have lengths = num_labels + 1
        print('Prediction list size = ', len(list(set(all_pred))))
        print('Target list size = ', len(list(set(all_target))))
        labels = [x for x in range(num_labels + 1)]

        print('Micro precision is %f' % precision_score(all_target, all_pred, labels= labels, average="micro"))
        print('Macro precision is %f' % precision_score(all_target, all_pred, labels= labels, average="macro"))
        print('Per class precision is\n', precision_score(all_target, all_pred, labels= labels, average=None))

        print('Micro recall is %f' % recall_score(all_target, all_pred, labels=labels, average="micro"))
        print('Macro recall is %f' % recall_score(all_target, all_pred, labels=labels, average="macro"))
        print('Per class recall is\n', recall_score(all_target, all_pred, labels=labels, average=None))

        print('Micro F1 score is %f' % f1_score(all_target, all_pred, labels=labels, average="micro"))
        print('Macro F1 score is %f' % f1_score(all_target, all_pred, labels=labels, average="macro"))
        print('Per class F1 score is\n', f1_score(all_target, all_pred, labels=labels, average=None))

        # for i in range(num_classes+1):
        #     print('%d %d %d'%(i, np.sum(np.equal(all_target, i)),\
        #                       np.sum(np.equal(all_pred, i))))

        # value = num_label means "no-rel"
        existing_relations = np.not_equal(all_target, num_labels)
        total_accuracy = np.mean(np.equal(all_pred, all_target))
        num_rel_correct = np.sum(existing_relations * np.equal(all_target, \
                                                               all_pred))
        accuracy_existing_relations = num_rel_correct / np.sum(existing_relations)
        return total_accuracy, accuracy_existing_relations


model = CopyEditor(EMBEDDING_SIZE, num_classes, copy=args.copy,
                   generate=args.generate)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

acc1, acc2 = accuracy(valdata, model)
print('Accuracy on val set = %f, Accuracy excluding norel=%f' % (acc1, acc2))
