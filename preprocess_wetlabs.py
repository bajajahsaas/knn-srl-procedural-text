import nltk
import json
import operator
import os
import numpy as np
import csv
import re
import random
import pickle
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
from bratreader.repomodel import RepoModel

train = 'WLP-Dataset/train'
test = 'WLP-Dataset/test'
val = 'WLP-Dataset/dev'

# r = RepoModel('wlpdata')

def get_words(annotation):
    return ' '.join([x.form for x in annotation.words])

def preprocess_linktype(linktype):
    # remove trailing numbers
    return re.sub('\d+$','',c)

def get_sentences(directory):
    r = RepoModel(directory)
    all_docs = []
    for fil in r.documents:
        doc = r.documents[fil]
        sentinfo = []
        for s in doc.sentences:
            sents_repl = []
            i = 0
            while i < len(s.words):
                if not len(s.words[i].annotations):
                    sents_repl.append(s.words[i].form)
                    i += 1
                else:
                    sents_repl.append('~~' + \
                                      list(s.words[i].annotations[0].labels.keys())[0]\
                                      + '~~')
                    i += len(s.words[i].annotations[0].words)
            sentinfo.append({'sentence': get_words(s),\
                                      'replaced':' '.join(sents_repl),\
                                        'relations' : [], 'key': s.key})
        for annotation in set(doc.annotations):
            for linktype in annotation.links:
                for tail in annotation.links[linktype]:
                    sent = annotation.words[0].sentkey
                    if sent != tail.words[0].sentkey:
                        continue
                    sentinfo[sent]['relations'].append({'head': {'text': get_words(annotation),
                                               'type': \
                                               list(annotation.labels.keys())[0],
                                               'spans': annotation.spans},
                                      'tail': {'text': get_words(tail),
                                               'type': \
                                               list(tail.labels.keys())[0],
                                               'spans': tail.spans},
                                      'relation_type': preprocess_linktype(linktype)})
        all_docs.extend(sentinfo)
    return all_docs


train_json = get_sentences(train)
test_json = get_sentences(test)
val_json = get_sentences(val)


with open('wetlabs_train.json', 'w') as f:
    json.dump(train_json, f, indent = 4)

with open('wetlabs_test.json', 'w') as f:
    json.dump(test_json, f, indent = 4)

with open('wetlabs_val.json', 'w') as f:
    json.dump(val_json, f, indent = 4)
