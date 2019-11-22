from annoy import AnnoyIndex
from matplotlib import pyplot as plt
import matplotlib
import pprint
import sys
import os
import pickle
import numpy as np
import json
matplotlib.rcParams['figure.dpi'] = 200

pp = pprint.PrettyPrinter(indent=4)
vectorizer_file = 'replaced_tfidf_small.pkl'
annoy_file = 'replaced_small.annoy'

with open(vectorizer_file, 'rb') as f:
    vect = pickle.load(f)

num_features = len(vect.get_feature_names())
t = AnnoyIndex(num_features, 'angular')
t.load(annoy_file)

with open('wetlabs_train.json', 'r') as f:
    train_data = json.load(f)
    train_data = train_data[:50]


with open('wetlabs_test.json', 'r') as f:
    test_data = json.load(f)

N = 100
K = 20

def get_relation_set(obj):
    relations_string = set()
    relations_type = set()
    relations_no_type = set()
    for relation in obj['relations']:
        relations_string.add((relation['head']['text'], relation['tail']['text'],
                          relation['relation_type']))
        relations_type.add((relation['head']['type'], relation['tail']['type'],
                          relation['relation_type']))
        relations_no_type.add(relation['relation_type'])
    return relations_string, relations_type, relations_no_type

recalls_string = {i:[] for i in range(1, K+1)}
recalls_type = {i:[] for i in range(1, K+1)}
recalls_no_type = {i:[] for i in range(1, K+1)}
hist_data = []
for s in test_data:
    print(s['sentence'])
    vector = vect.transform([s['replaced']]).toarray()[0]
    nns = t.get_nns_by_vector(vector, K)
    if len(s['relations']) == 0:
        continue
    found_relations_string = set()
    found_relations_type = set()
    found_relations_no_type = set()
    relations_string, relations_type, relations_no_type = get_relation_set(s)
    for ind, j in enumerate(nns):
        r1, r2, r3 = get_relation_set(train_data[j])
        found_relations_type |= r2
        found_relations_string |= r1
        found_relations_no_type |= r3
        v1 = float(len(found_relations_string & \
                       relations_string))/float(len(relations_string))
        v2 = float(len(found_relations_type & \
                       relations_type))/float(len(relations_type))
        v3 = float(len(found_relations_no_type & \
                       relations_no_type))/float(len(relations_no_type))
        recalls_string[ind+1].append(v1)
        recalls_type[ind+1].append(v2)
        recalls_no_type[ind+1].append(v3)


plt.figure(1)
numbins = 20
plt.hist(recalls_type[5], numbins)
plt.show()

for key in recalls_string:
    recalls_string[key] = np.mean(recalls_string[key])



for key in recalls_type:
    recalls_type[key] = np.mean(recalls_type[key])


for key in recalls_no_type:
    recalls_no_type[key] = np.mean(recalls_no_type[key])


plt.figure(2)
string_recalls = [recalls_string[i] for i in range(1,K+1)]
type_recalls = [recalls_type[i] for i in range(1,K+1)]
no_type_recalls = [recalls_no_type[i] for i in range(1,K+1)]
print(no_type_recalls)
xaxis = [i for i in range(1, K+1)]

with open('recall_data.pkl', 'wb') as f:
    pickle.dump(zip(xaxis, string_recalls, type_recalls, no_type_recalls), f) 


plt.plot(xaxis, string_recalls, label = 'Recals for lexical match')
plt.plot(xaxis, type_recalls, label = 'Recals for relation and entity type match')
plt.plot(xaxis, no_type_recalls, label = 'Recals for only relation type match')
plt.xticks(np.arange(1,K+1))
plt.yticks(np.arange(0,1,0.05))
plt.legend()
plt.xlabel('Number of retrieved samples')
plt.ylabel('Recall')
plt.show()
