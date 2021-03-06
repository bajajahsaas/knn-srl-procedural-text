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

# 10967 line number
N = 20
k = 5
l = []
for i in np.random.choice(list(range(len(sent_data))), size=N, replace=False):
    s = sent_data[i]
    if len(s) < 70:
        continue
    nns = t.get_nns_by_item(i, k)
    l.append(i)
    print('Sentence index: %s' % i)
    print('Sentence: %s'%s)
    print('Orig_sentence: %s'%orig_sents[i])

    print('Nearest Neighbors:')
    pp.pprint([sent_data[x] for x in nns[1:]])
    print('Original sentences:')
    pp.pprint([orig_sents[x] for x in nns[1:]])

print(l)
