import sys
import numpy as np
import os
from annoy import AnnoyIndex
import pickle
import re
'''
produces file like 
[{
    query_relations: N x 2 x 3 array (N relations, head-tail, token spans + type)
    context_relations: [N x 2 x 3] list of size K
    query_labels : N
    context_labels: [N]xK
    query_tokens : list of tokens
    context_tokens : [list of tokens for each context sentence] list of size K
}
...]
    
'''
def get_relations_entity_spans(s):
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
    relations_list = []
    for h, t, label in all_rels:
        head = [s['entities'][h][2][0], s['entities'][h][2][1], \
                    ent_dic[s['entities'][h][1]]]
        tail = [s['entities'][t][2][0], s['entities'][t][2][1], \
                    ent_dic[s['entities'][t][1]]]
        relations_list.append([head, tail])
    labels = [x[2] for x in all_rels]
    return np.asarray(s['sent_tokens']), np.asarray(relations_list), np.asarray(labels)




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
K = 5 # 5 nearest neighbors


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
    relations = list(relations)
    entity_types = list(entity_types)
    with open('relations.txt', 'w') as f:
        f.write('\n'.join(relations))
    with open('entity_types.txt', 'w') as f:
        f.write('\n'.join(entity_types))
print(relations)
print(len(relations))
relations.append('No relation')
rel_dic = {r:i for i, r in enumerate(relations)}
print(entity_types)
print(len(entity_types))
ent_dic = {e:i for i, e in enumerate(entity_types)}


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

    num_context = 0
    context_labels = []
    context_relations = []
    context_tokens = []
    for nn_id in nns:
        cs = data_src[nn_id]
        tokens, relations, labels= get_relations_entity_spans(cs)
        if tokens is not None:
            num_context += 1
            context_labels.append(labels)
            context_relations.append(relations)
            context_tokens.append(tokens)
    
    tokens, relations, labels = get_relations_entity_spans(s)
    if tokens is None:
        continue
    dataset.append({
                        'query_sent' : s['sentence'],
                        'context_sents' : [data_src[x]['sentence'] for x in \
                                           nns],
                        'query_tokens' : tokens,
                        'context_tokens': context_tokens,
                        'query_relations' : relations,
                        'context_relations' : context_relations,
                        'query_labels': labels,
                        'context_labels' : context_labels
                    })

with open(sys.argv[5], 'wb') as f:
    pickle.dump(dataset, f)
