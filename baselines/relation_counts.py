import json
import sys
import operator
from sklearn.metrics import precision_score, recall_score, f1_score

train = sys.argv[1]
val = sys.argv[2]
test = sys.argv[3]

with open(train, 'r') as f:
    data_train = json.load(f)

with open(val, 'r') as f:
    data_val = json.load(f)

with open(test, 'r') as f:
    data_test = json.load(f)


def getrelationcounts(data, type):
    relations_counts = {}
    for datum in data:
        for r in datum['relations']:
            relation_type = r['relation_type']
            if relation_type not in relations_counts:
                relations_counts[relation_type] = 1
            else:
                relations_counts[relation_type] += 1

    sorted_keys = sorted(relations_counts.keys())
    with open('relation_counts_' + type + '.txt', 'w') as f:
        f.write('Relation\tCount\n')
        for k in sorted_keys:
            f.write('%s\t%s\n' % (k, relations_counts[k]))


getrelationcounts(data_train, 'train')
getrelationcounts(data_val, 'val')
getrelationcounts(data_test, 'test')
