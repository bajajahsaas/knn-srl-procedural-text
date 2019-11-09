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
r = RepoModel('wlpdata')

print('Preprocessing')

documents = list(r.documents.keys())
random.shuffle(documents)
trainlen = int(len(documents) * 0.8)
traindata = documents[:trainlen]
testdata = documents[trainlen:]

corrected_ann = 0
docs_corrected_ann = 0

def get_annotation_map(doc):
    annotation_map = dict()
    for annotation in set(doc.annotations):
        text = get_words(annotation)
        entity_type = list(annotation.labels.keys())[0]
        entity_spans = annotation.spans # list of lists (having two values each). Length of this list gives total spans

        if text not in annotation_map:
            annotation_map[text] = dict()
            annotation_map[text][entity_type] = 1

        elif entity_type not in annotation_map[text]:
            annotation_map[text][entity_type] = 1

        else:
            annotation_map[text][entity_type] +=  1
    return annotation_map

def get_words(annotation):
    return ' '.join([x.form for x in annotation.words])

def get_sentences(docs):
    global corrected_ann, docs_corrected_ann

    all_docs = []
    for fil in docs:
        previous_corrected_ann = corrected_ann

        doc = r.documents[fil]
        annotation_map = get_annotation_map(doc)
        sentinfo = []
        for s in doc.sentences:
            sents_repl = []
            i = 0
            while i < len(s.words):
                if not len(s.words[i].annotations):
                    text = s.words[i].form
                    if text not in annotation_map:
                        # Handles only one word missing annotations, these might also occur when mention spans > 1 word
                        sents_repl.append(s.words[i].form)
                    else:
                        corrected_ann += 1
                        annotation_dict = annotation_map[text]

                        if len(list(annotation_dict.keys())) > 1:
                            # Multiple type of annotation for same word
                            print(text, annotation_dict)

                        # Find entity_type with max value (size of entity_spans list). Bcoz many words have duplicate annotations in same doc
                        # Resolve by finding majority
                        entity_type = max(annotation_dict.items(), key=operator.itemgetter(1))[0]

                        sents_repl.append('~~' + entity_type + '~~')

                    i += 1
                else:
                    sents_repl.append('~~' + \
                                      list(s.words[i].annotations[0].labels.keys())[0]\
                                      + '~~')
                    i += len(s.words[i].annotations[0].words)
            sentinfo.append({'sentence': get_words(s),\
                                      'replaced':' '.join(sents_repl),\
                                        'relations' : [], 'key': s.key})

        if corrected_ann > previous_corrected_ann:
            previous_corrected_ann = corrected_ann
            docs_corrected_ann += 1

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
                                      'relation_type': linktype})
        all_docs.extend(sentinfo)
    return all_docs


train_json = get_sentences(traindata)
test_json = get_sentences(testdata)

print('Corrected annotations', corrected_ann)
print('Corrected Docs', docs_corrected_ann)

with open('wetlabs_train.json', 'w') as f:
    json.dump(train_json, f, indent = 4)

with open('wetlabs_test.json', 'w') as f:
    json.dump(test_json, f, indent = 4)



