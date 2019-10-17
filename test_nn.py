from annoy import AnnoyIndex
import pprint
import sys
import os
import pickle
import numpy as np

pp = pprint.PrettyPrinter(indent=4)
vectorizer_file = sys.argv[1]
annoy_file = sys.argv[2]
sentences = sys.argv[3]
sent2 = sys.argv[4] # just give same file twice if not needed

with open(sentences, 'r') as f:
    sent_data = f.read().splitlines()

with open(sent2, 'r') as f:
    orig_sents = f.read().splitlines()

with open(vectorizer_file, 'rb') as f:
    vect = pickle.load(f)

num_features = len(vect.get_feature_names())
t = AnnoyIndex(num_features, 'angular')
t.load(annoy_file)

N = 5
for i in np.random.choice(list(range(len(sent_data))), size=5, replace = False): 
    s = sent_data[i]
    nns = t.get_nns_by_item(i, 10)
    
    print('Sentence: %s'%s)
    print('Orig_sentence: %s'%orig_sents[i])
    
    print('Nearest Neighbors:')
    pp.pprint([orig_sents[x] for x in nns[1:]])

