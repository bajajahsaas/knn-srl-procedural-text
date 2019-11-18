import json
import sys
import operator
from sklearn.metrics import precision_score, recall_score, f1_score

src = sys.argv[1]
tgt = sys.argv[2]

with open(src, 'r') as f:
    data_src = json.load(f)


with open(tgt, 'r') as f:
    data_tgt = json.load(f)

relations_counts = {}
labels = set()
for datum in data_src:
    for r in datum['relations']:
        ht_type = (r['head']['type'], r['tail']['type'])
        if ht_type not in relations_counts:
            relations_counts[ht_type] = {}
        if r['relation_type'] not in relations_counts[ht_type]:
            relations_counts[ht_type][r['relation_type']] = 1
        else:
            relations_counts[ht_type][r['relation_type']] += 1
        labels.add(r['relation_type'])

majority_class = {}
k = 6
for ht in relations_counts:
    sorted_rels = sorted(relations_counts[ht].items(), key=operator.itemgetter(1))
    rels = []
    for i in range(1, min(len(sorted_rels) + 1, k)):
        rels.append(sorted_rels[-i][0])
    majority_class[ht] = rels

with open('majority_results.txt', 'w') as f:
    f.write('Head\tTail\tClass\n')
    for k,v in majority_class.items():
        f.write('%s\t%s\t%s\n' % (k[0], k[1], ','.join(v)))

labels = list(labels)
print(majority_class)
print(labels)

all_target = []
all_pred = []

for datum  in data_tgt:
    relset = set()
    all_entities = list(set([(r['head']['text'], r['head']['type']) for r in \
                         datum['relations']]+[(r['tail']['text'],r['tail']['type']) \
                                                        for r in datum['relations']]))
    entities = [x for x in all_entities if x[0] in datum['sentence']]
    entity_dic = {e:i for i, e in enumerate(entities)}
    relations = []
    for r in datum['relations']:
        if (r['head']['text'], r['head']['type']) in entity_dic and (r['tail']['text'], r['tail']['type']) in entity_dic:
            relset.add((entity_dic[(r['head']['text'],r['head']['type'])],\
                   entity_dic[(r['tail']['text'],r['tail']['type'])]))
            relations.append((r['head']['type'], r['tail']['type'], r['relation_type']))

    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j or (i,j) in relset:
                continue
            relations.append((entities[i][1],entities[j][1],'No relation'))

    all_target.extend([x[2] for x in relations])
    all_pred.extend([majority_class.get((x[0],x[1]), 'No relation')[0] for x in \
                     relations])


print('Micro precision is ', str(round(precision_score(all_target, all_pred, labels= labels, average="micro"),2)))
print('Macro precision is ', str(round(precision_score(all_target, all_pred, labels= labels, average="macro"),2)))
print('Per class precision is\n', str(round(precision_score(all_target, all_pred, labels= labels, average=None),2)))

print('Micro recall is ', str(round(recall_score(all_target, all_pred, labels=labels, average="micro"),2)))
print('Macro recall is ', str(round(recall_score(all_target, all_pred, labels=labels, average="macro"),2)))
print('Per class recall is\n', str(round(recall_score(all_target, all_pred, labels=labels, average=None),2)))

print('Micro F1 score is ', str(round(f1_score(all_target, all_pred, labels=labels, average="micro"),2)))
print('Macro F1 score is ', str(round(f1_score(all_target, all_pred, labels=labels, average="macro"),2)))
print('Per class F1 score is\n', str(round(f1_score(all_target, all_pred, labels=labels, average=None),2)))
