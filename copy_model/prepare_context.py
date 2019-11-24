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
        return None, None, None, None, None, None, None
    for i in range(len(s['entities'])):
        for j in range(len(s['entities'])):
            if i == j or (i, j) in relset:
                continue
            norels.append((i,j,'No relation'))
    all_rels = [(a,b,rel_dic[re.sub('\d+$','',c)]) for a,b,c in s['relations'] + norels]
    head = np.stack([s['entities'][x[0]][2] for x in all_rels])
    head_text = [s['entities'][x[0]][0] for x in all_rels]
    head_type = np.stack([ent_dic[s['entities'][x[0]][1]] for x in all_rels])
    tail = np.stack([s['entities'][x[1]][2] for x in all_rels])
    tail_text = [s['entities'][x[1]][0] for x in all_rels]
    tail_type = np.stack([ent_dic[s['entities'][x[1]][1]] for x in all_rels])
    labels = np.stack([x[2] for x in all_rels])
    return head_text, head, head_type, tail_text, tail, tail_type, labels




annoy_file = sys.argv[1]
vectorizer_file = sys.argv[2]
bert_data = sys.argv[3]
bert_data_target = sys.argv[4]

with open(bert_data, 'rb') as f:
    data_src = pickle.load(f)

with open(bert_data_target, 'rb') as f:
    data_tgt = pickle.load(f)

with open(vectorizer_file, 'rb') as f:
    vect = pickle.load(f)

t = AnnoyIndex(len(vect.get_feature_names()), 'angular')
t.load(annoy_file)

bad_relations = set(['Misc-Link'])


if os.path.exists('relations.txt') and os.path.exists('entity_types.txt'):
    with open('relations.txt', 'r') as f:
        relations = f.read().splitlines()
    with open('entity_types.txt', 'r') as f:
        entity_types = f.read().splitlines()
else:
    relations = set()
    entity_types = set()
    for s in data_src:
        for r in s['relations']:
            print(r)
            relation_without_trailing = re.sub('\d+$','',r[2]) 
            relations.add(relation_without_trailing)
        for e in s['entities']:
            entity_types.add(e[1])
    relations = list(relations - bad_relations)
    entity_types = list(entity_types)
    with open('relations.txt', 'w') as f:
        f.write('\n'.join(relations))
    with open('entity_types.txt', 'w') as f:
        f.write('\n'.join(entity_types))
print(relations)
print(len(relations))
relations.append('No relation')
rel_dic = {r:i for i, r in enumerate(relations)}
for r in bad_relations:
    rel_dic[r] = len(relations) -1 # norelation
print(entity_types)
print(len(entity_types))
ent_dic = {e:i for i, e in enumerate(entity_types)}

nns = [10]
# choice for nearest neighbors

for K in nns:
    dataset = []
    cnt = 0
    for s in data_tgt:
        print('%d / %d done'%(cnt, len(data_tgt)))
        cnt += 1
        vector = vect.transform([s['replaced']]).toarray()[0]
        if bert_data == bert_data_target: # on training data, we are bound to find
                                          # same sentence, so we exclude the top
                                          # match
            nns = t.get_nns_by_vector(vector, K+1)[1:] #1st one will be the same
                                                    # sentence
        else: # on held out data, we needn't exclude the top match
            nns = t.get_nns_by_vector(vector, K+1)

        context_head = []
        context_head_text = []
        context_head_type = []
        context_tail = []
        context_tail_text = []
        context_tail_type = []
        context_label = []
        num_context = 0
        for nn_id in nns:
            cs = data_src[nn_id]
            head_text, head, head_type, tail_text, tail, tail_type, labels = get_relations_embeddings(cs)
            if head is not None:
                num_context += 1
                context_head.append(head)
                context_head_text.extend(head_text)
                context_tail.append(tail)
                context_tail_text.extend(tail_text)
                context_label.append(labels)
                context_head_type.append(head_type)
                context_tail_type.append(tail_type)

        if num_context > 0:
            context_head = np.concatenate(context_head, axis=0)
            context_tail = np.concatenate(context_tail, axis=0)
            context_label = np.concatenate(context_label, axis = 0)
            context_head_type = np.concatenate(context_head_type, axis = 0)
            context_tail_type = np.concatenate(context_tail_type, axis = 0)
        else:
            context_head, context_head_type, context_tail, context_tail_type, \
                    context_label = None, None, None, None, None

        query_head_text, query_head, query_head_type, query_tail_text, query_tail, query_tail_type, query_labels = get_relations_embeddings(s)
        if query_head is None:
            continue
        dataset.append({
                            'query_sent' : s['sentence'],
                            'context_sents' : [data_src[x]['sentence'] for x in \
                                               nns],
                            'query_head': query_head,
                            'query_head_text' : query_head_text,
                            'query_head_type': query_head_type,
                            'query_tail': query_tail,
                            'query_tail_text' : query_tail_text,
                            'query_tail_type': query_tail_type,
                            'query_labels': query_labels,
                            'context_head' : context_head,
                            'context_head_text' : context_head_text,
                            'context_head_type': context_head_type,
                            'context_tail' : context_tail,
                            'context_tail_text' : context_tail_text,
                            'context_tail_type': context_tail_type,
                            'context_labels' : context_label
                        })
    fil = sys.argv[5]
    st = fil[:len(fil) - 4] + "_K" + str(K) + fil[len(fil) - 4:]
    with open(st, 'wb') as f:
        pickle.dump(dataset, f)
