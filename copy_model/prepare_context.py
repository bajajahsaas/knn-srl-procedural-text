import sys
import numpy as np
import os
from annoy import AnnoyIndex
import pickle
import re

def get_relations_embeddings(s):
    # Get all relations including no relation
    relset = set([(a,b) for a, b, _ in s['relations']])
    norels = []
    if len(s['entities']) <= 1:
        return None, None, None
    for i in range(len(s['entities'])):
        for j in range(len(s['entities'])):
            if i == j or (i, j) in relset:
                continue
            norels.append((i,j,'No relation'))
    all_rels = [(a,b,rel_dic[re.sub('\d+$','',c)]) for a,b,c in s['relations'] + norels]
    head = np.stack([s['entities'][x[0]][1] for x in all_rels])
    tail = np.stack([s['entities'][x[1]][1] for x in all_rels])
    labels = np.stack([x[2] for x in all_rels])
    return head, tail, labels




annoy_file = sys.argv[1]
vectorizer_file = sys.argv[2]
bert_data = sys.argv[3]

with open(bert_data, 'rb') as f:
    data = pickle.load(f)

with open(vectorizer_file, 'rb') as f:
    vect = pickle.load(f)

t = AnnoyIndex(len(vect.get_feature_names()), 'angular')
t.load(annoy_file)
K = 5 # 5 nearest neighbors

relations = set()
for s in data:
    for r in s['relations']:
        # print(r)
        relation_without_trailing = re.sub('\d+$','',r[2]) 
        relations.add(relation_without_trailing)
relations = list(relations)
print(relations)
print(len(relations))
relations.append('No relation')
rel_dic = {r:i for i, r in enumerate(relations)}


dataset = []
cnt = 0
for s in data:
    print('%d / %d done'%(cnt, len(data)))
    cnt += 1
    vector = vect.transform([s['replaced']]).toarray()[0]
    nns = t.get_nns_by_vector(vector, K+1)[1:] #1st one will be the same
                                                # sentence
    context_head = []
    context_tail = []
    context_label = []
    num_context = 0
    for nn_id in nns:
        cs = data[nn_id]
        head, tail, labels = get_relations_embeddings(cs)
        if head is not None:
            num_context += 1
            context_head.append(head)
            context_tail.append(tail)
            context_label.append(labels)
    
    if num_context > 0:
        context_head = np.concatenate(context_head, axis=0)
        context_tail = np.concatenate(context_tail, axis=0)
        context_label = np.concatenate(context_label, axis = 0)
    else:
        context_head, context_tail, context_label = None, None, None

    query_head, query_tail, query_labels = get_relations_embeddings(s)
    
    dataset.append({
                        'query_head': query_head,
                        'query_tail': query_tail,
                        'query_labels': query_labels,
                        'context_head' : context_head,
                        'context_tail' : context_tail,
                        'context_labels' : context_label
                    })

with open('dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
