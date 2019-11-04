import nltk
import operator
import os
import numpy as np
import csv
import re
import pickle
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import convert_bert_original_tf_checkpoint_to_pytorch, BertForTokenClassification, BertTokenizer, \
    BertForSequenceClassification

'''
IMP Links:
https://github.com/BaderLab/saber/issues/135
https://github.com/dmis-lab/biobert/issues/26
https://colab.research.google.com/drive/1RhmL0BqNe52FEbdSyLpkfVuCZxE7b5ke#scrollTo=4kzMl3rQJjgV
https://towardsdatascience.com/nlp-extract-contextualized-word-embeddings-from-bert-keras-tf-67ef29f60a7b
https://github.com/naver/biobert-pretrained/
https://github.com/dmis-lab/biobert/issues/23
https://github.com/hanxiao/bert-as-service
'''


BERT_BASE_DIR='../../biobert_v1.1_pubmed/'
file = BERT_BASE_DIR + 'pytorch_model.bin'

if os.path.exists(file):
    print("PyTorch version of bioBERT found")
else:
    print('Convert tf checkpoint to pyTorch')
    convert_bert_original_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(BERT_BASE_DIR + 'model.ckpt-1000000', BERT_BASE_DIR + 'bert_config.json' , BERT_BASE_DIR + 'pytorch_model.bin')


BIOBERT_DIR = '../../biobert'
# Rename bert_config.json to config.json

model = BertForSequenceClassification.from_pretrained(BIOBERT_DIR)
tokenizer = BertTokenizer.from_pretrained(BIOBERT_DIR, do_lower_case=False)


