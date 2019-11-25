import nltk
import json
import operator
import os
import numpy as np
import csv
import re
import pickle
from annoy import AnnoyIndex
import random
from sklearn.feature_extraction.text import TfidfVectorizer


with open('wetlabs_train.json', 'r') as f:
    data = json.load(f)

    data = data[:50]
    replaced = []
    originals = []
    for k in data:
        originals.append(k["sentence"])
        replaced.append(k["replaced"])

    with open('replaced_sentences_small.txt', 'w') as f:
        f.write('\n'.join(replaced))

    with open('original_sentences_small.txt', 'w') as f:
        f.write('\n'.join(originals))

    print('Finished Preprocessing')

def build_annoy_tfidf(sentences):
    print('Getting tfidf vectors')
    # Get TF IDF representation for each sentence
    vectorizer = TfidfVectorizer(ngram_range = (1, 4), min_df = 2, max_df = 1.0)
    vectorizer.fit(sentences)
    X = vectorizer.transform(sentences)
    
    # build annoy index
    num_features = len(vectorizer.get_feature_names())
    t = AnnoyIndex(num_features, "angular") # NN with cosine distance
    print('Inserting into annoy')
    for i, sent in enumerate(X):
        t.add_item(i, sent.toarray()[0])
        if i%100 == 0:
            print('%d/%d done'%(i, len(sentences)))
    print('Building annoy')
    t.build(10)
    return vectorizer, t


print('Building annoy index for replaced sentences')
# Representations of replaced sentences
v_rep, ann_rep = build_annoy_tfidf([x['replaced'] for x in data])
ann_rep.save('replaced_small.annoy')
with open('replaced_tfidf_small.pkl', 'wb') as f:
    pickle.dump(v_rep, f)


print('Building annoy index for original documents')
# Representations of original sentences
v_ori, ann_ori = build_annoy_tfidf([x['sentence'] for x in data])
ann_ori.save('original_small.annoy')
with open('original_tfidf_small.pkl', 'wb') as f:
    pickle.dump(v_ori, f)
