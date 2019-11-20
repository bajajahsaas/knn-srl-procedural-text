from annoy import AnnoyIndex
import pprint
import sys
import os
import pickle
import numpy as np
import json

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
K = 10

def get_relation_set(obj):
    relations_string = set()
    relations_type = set()
    for relation in obj['relations']:
        relations_string.add((relation['head']['text'], relation['tail']['text'],
                          relation['relation_type']))
        relations_type.add((relation['head']['type'], relation['tail']['type'],
                          relation['relation_type']))
    return relations_string, relations_type



for i in np.random.choice(list(range(len(test_data))), size=N, replace = False): 
    s = test_data[i]
    vector = vect.transform([s['replaced']]).toarray()[0]
    nns = t.get_nns_by_vector(vector, K)
    
    print('Sentence: %s'%s['replaced'])
    print('Orig_sentence: %s'%s['sentence'])
    
    print('Nearest Neighbors:')
    pp.pprint([train_data[x]['replaced'] for x in nns])
    print('Original sentences:')
    pp.pprint([train_data[x]['sentence'] for x in nns])
    
    if len(s['relations']) == 0:
        continue
    found_relations_string = set()
    found_relations_type = set()
    for j in nns:
        r1, r2 = get_relation_set(train_data[j])
        found_relations_type |= r2
        found_relations_string |= r1

    relations_string, relations_type = get_relation_set(s)
    print('portion of lexical relations overlap =%f'%(float(len(relations_string & found_relations_string))/float(len(relations_string))))
    print('portion of type relations overlap =%f'%(float(len(relations_type & found_relations_type))/float(len(relations_type))))

