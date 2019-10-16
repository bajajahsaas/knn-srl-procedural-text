import nltk
import operator
import os
import numpy as np
import csv
import re
import pickle
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer


print('Preprocessing')
dire = 'wlpdata'

files = list(set([x.split('.')[0] for x in os.listdir(dire) if
                  re.match('protocol_[\d]*.(ann|txt)', x) ]))

def process_file(protocol, entities):
    entity_list = []
    with open(entities, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        for line in reader:
           if line[0].startswith('T'):
                tag = line[1].split()[0]
                occurences = line[1][len(tag):].split(';')
                tag = '~' + tag + '~'
                for occurence in occurences:
                    split = occurence.split()
                    start = int(split[0])
                    end = int(split[1])
                    term = line[2]
                    entity_list.append((start, end, tag))
    entity_list = sorted(entity_list, key = operator.itemgetter(0)) # sort by starting index 
    lastindex = 0
    newstring = ''
    for start, end, tag in entity_list:
            newstring += protocol[lastindex:start] + ' ' + tag+' '
            lastindex = end + 1
    newstring += protocol[lastindex:]
    return zip(newstring.splitlines(), protocol.splitlines())

sentences = []
for i, fil in enumerate(files):
    with open(os.path.join(dire, '%s.txt'%fil), 'r') as f:
        protocol = f.read()
    sentences.extend(process_file(protocol, os.path.join(dire, '%s.ann'%fil)))
    if i%10 == 0:
        print('%d/%d done'%(i, len(files)))

replaced, originals = zip(*sentences)
with open('replaced_sentences.txt', 'w') as f:
    f.write('\n'.join(replaced))

with open('original_sentences.txt', 'w') as f:
    f.write('\n'.join(originals))
print('Finished Preprocessing')


def build_annoy_tfidf(sentences):
    print('Getting tfidf vectors')
    # Get TF IDF representation for each sentence
    vectorizer = TfidfVectorizer(ngram_range = (1, 3))
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
v_rep, ann_rep = build_annoy_tfidf(replaced)
ann_rep.save('replaced.annoy')
with open('replaced_tfidf.pkl', 'wb') as f:
    pickle.dump(v_rep, f)


print('Building annoy index for original documents')
# Representations of original sentences
v_ori, ann_ori = build_annoy_tfidf(originals)
ann_ori.save('original.annoy')
with open('original_tfidf.pkl', 'wb') as f:
    pickle.dump(v_ori, f)


