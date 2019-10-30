import nltk
import json
import operator
import os
import numpy as np
import csv
import re
import pickle
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer


with open('wetlabs_train.json', 'r') as f:
    data = json.load(f)

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
ann_rep.save('replaced.annoy')
with open('replaced_tfidf.pkl', 'wb') as f:
    pickle.dump(v_rep, f)


print('Building annoy index for original documents')
# Representations of original sentences
v_ori, ann_ori = build_annoy_tfidf([x['sentence'] for x in data])
ann_ori.save('original.annoy')
with open('original_tfidf.pkl', 'wb') as f:
    pickle.dump(v_ori, f)
