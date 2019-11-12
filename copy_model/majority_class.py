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
for ht in relations_counts:
    sorted_rels = sorted(relations_counts[ht].items(), key=operator.itemgetter(1))
    majority_class[ht] = sorted_rels[-1][0]

with open('majority_results.txt', 'w') as f:
    f.write('Head\tTail\tClass\n')
    for it in majority_class.items():
        f.write('%s\t%s\t%s\n'%(it[0][0],it[0][1],it[1]))
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
    all_pred.extend([majority_class.get((x[0],x[1]), 'No relation') for x in \
                     relations])


print('Micro precision is %f' % precision_score(all_target, all_pred, labels= labels, average="micro"))
print('Macro precision is %f' % precision_score(all_target, all_pred, labels= labels, average="macro"))
print('Per class precision is\n', precision_score(all_target, all_pred, labels= labels, average=None))

print('Micro recall is %f' % recall_score(all_target, all_pred, labels=labels, average="micro"))
print('Macro recall is %f' % recall_score(all_target, all_pred, labels=labels, average="macro"))
print('Per class recall is\n', recall_score(all_target, all_pred, labels=labels, average=None))

print('Micro F1 score is %f' % f1_score(all_target, all_pred, labels=labels, average="micro"))
print('Macro F1 score is %f' % f1_score(all_target, all_pred, labels=labels, average="macro"))
print('Per class F1 score is\n', f1_score(all_target, all_pred, labels=labels, average=None))
