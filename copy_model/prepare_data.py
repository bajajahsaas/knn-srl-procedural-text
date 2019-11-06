import json
import numpy as np
import pickle
import sys
import torch
from biobert import *

# for each sentence compute embeddings of all entities 

traindata = sys.argv[1]

usebiobert=False
if sys.argv[2] == 'biobert':
    usebiobert = True

with open(traindata, 'r') as f:
    data = json.load(f)

if usebiobert:
    tokenizer, model = getbiobertmodel()
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

# else default ones imported from biobert

embedded_data = []

def find_indices(lis, sublist):
    for i in range(len(lis) - len(sublist) +1):
        if lis[i:i+len(sublist)] == sublist:
            return i
    return -1

count = 0
for sent in data:
    print('%d / %d done'%(count, len(data)))
    count += 1
    all_entities = list(set([r['head']['text'] for r in \
                         sent['relations']]+[r['tail']['text'] \
                                                        for r in sent['relations']]))
    entities = [x for x in all_entities if x in sent['sentence']]
    entity_dic = {e:i for i, e in enumerate(entities)}
    sent_text = sent['sentence']
    bert_tokens_entities = [tokenizer.encode(e, add_special_tokens=False) for e in entities]
    bert_tokens_sentence = tokenizer.encode(sent_text, add_special_tokens=True)
    entity_spans = []
    filtered_entities = []
    for i, tokenized_entity in enumerate(bert_tokens_entities):
        start = find_indices(bert_tokens_sentence, tokenized_entity)
        # print(start)
        if start == -1:
            print(bert_tokens_sentence)
            print(tokenized_entity)
            print(sent['sentence'])
            print(entities[i])
        end = start + len(tokenized_entity) -1
        entity_spans.append((start, end))
    with torch.no_grad():
        bert_embeddings = \
            model(torch.tensor([bert_tokens_sentence]))[0].squeeze(0).numpy() 
    entity_embeddings = [np.mean(bert_embeddings[start:end+1], axis=0) for \
                            start, end in entity_spans]
    relations = []
    for x in sent['relations']:
        if x['head']['text'] in entity_dic and x['tail']['text'] in entity_dic:
            relations.append((entity_dic[x['head']['text']], entity_dic[x['tail']['text']],\
                    x['relation_type']))
    embedded_data.append({'entities': list(zip(entities, entity_embeddings)),
                          'relations' : relations, 'sentence' :
                          sent['sentence'], 'replaced' : sent['replaced']})

with open('train_embedding_data.pkl', 'wb') as f:
    pickle.dump(embedded_data, f)
