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
k = 5
ans = 0.0
count = 100
for iter in range(count):
    inner_mean = 0.0
    for i in np.random.choice(list(range(len(sent_data))), size=N, replace=False):
        s = sent_data[i]

        nns = t.get_nns_by_item(i, k, include_distances=True)
        # if include_distances = True, this returns a tuple of (item, distance)
        inner_mean += np.mean(nns[1])

    inner_mean /= N
    ans += inner_mean
    print('Mean distance after iter', iter + 1, ans / (iter + 1))
    # Annoy uses Euclidean distance of normalized vectors for its angular distance, which for two vectors u, v is equal to sqrt(2(1-cos(u, v)))
    # Min value = 0, Max value = root(2)
# Final Mean distance after 100  iterations ~ 0.6
print('Final Mean distance after', count, 'iterations = ', ans/count)
