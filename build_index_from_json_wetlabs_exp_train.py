import sys

import nltk
import json
import operator
import os
import numpy as np
import csv
import re
import pickle

import torch
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn

from copy_model.biobert import getscibertmodel

def build_annoy_tfidf(sentences):
    print('Getting tfidf vectors')
    # Get TF IDF representation for each sentence
    vectorizer = TfidfVectorizer(ngram_range=(1, 4), min_df=2, max_df=1.0)
    vectorizer.fit(sentences)
    X = vectorizer.transform(sentences)

    # build annoy index
    num_features = len(vectorizer.get_feature_names())
    t = AnnoyIndex(num_features, "angular")  # NN with cosine distance
    print('Inserting into annoy')
    for i, sent in enumerate(X):
        t.add_item(i, sent.toarray()[0])
        if i % 100 == 0:
            print('%d/%d done' % (i, len(sentences)))
    print('Building annoy')
    t.build(10)
    return vectorizer, t


def build_annoy_bert(sentences):
    tokenizer, model = getscibertmodel()
    print('Loaded scibert model')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X = {}
    for iter, sent in enumerate(sentences):
        bert_tokens_sentence = tokenizer.encode(sent, add_special_tokens=True)
        with torch.no_grad():
            bert_embeddings = \
                model(torch.tensor([bert_tokens_sentence]).to(device))[0].squeeze(0)
            f_emb_avg = torch.mean(bert_embeddings, axis=0).cpu().numpy()
            X[sent] = f_emb_avg

        if iter % 1000 == 0:
            print("BERT ready for ", iter, " sentences")

    # build annoy index
    num_features = 768
    t = AnnoyIndex(num_features, "angular")  # NN with cosine distance
    print('Inserting into annoy')
    for i, sent in enumerate(X):
        t.add_item(i, X[sent])
        if i % 100 == 0:
            print('%d/%d done' % (i, len(sentences)))
    print('Building annoy')
    t.build(10)
    return t


percentage = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6]

for per in percentage:
   with open('wetlabs_train' + str(per) + '.json', 'r') as f:
       data = json.load(f)
       # replaced = []
       # originals = []
       # for k in data:
       #     originals.append(k["sentence"])
       #     replaced.append(k["replaced"])
       #
       # with open('replaced_sentences' + str(per) + '.txt', 'w') as f:
       #     f.write('\n'.join(replaced))
       #
       # with open('original_sentences' + str(per) + '.txt', 'w') as f:
       #     f.write('\n'.join(originals))
       #
       # print('Finished Preprocessing')

       print('Finished Preprocessing')

   if sys.argv[1] == 'scibert':
       print('Using BERT for retriever')
       print('Building annoy index for original documents: ', str(per))
       # Representations of original sentences
       ann_ori = build_annoy_bert([x['sentence'] for x in data])
       ann_ori.save('original' + str(per) + '.annoy')

   else:
       print('Using tf-idf for retriever')
       print('Building annoy index for replaced sentences ', str(per))
       # Representations of replaced sentences
       v_rep, ann_rep = build_annoy_tfidf([x['replaced'] for x in data])
       ann_rep.save('replaced' + str(per) + '.annoy')
       with open('replaced_tfidf' + str(per) + '.pkl', 'wb') as f:
           pickle.dump(v_rep, f)

       print('Building annoy index for original documents ', str(per))
       # Representations of original sentences
       v_ori, ann_ori = build_annoy_tfidf([x['sentence'] for x in data])
       ann_ori.save('original' + str(per) + '.annoy')
       with open('original_tfidf' + str(per) + '.pkl', 'wb') as f:
           pickle.dump(v_ori, f)
