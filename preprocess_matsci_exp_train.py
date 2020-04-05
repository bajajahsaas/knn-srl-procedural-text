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

from copy_model.biobert import getscibertmodel
from bratreader.repomodel import RepoModel
from random import sample

with open('materials_train.json', 'r') as f:
    train_json = json.load(f)

percentage = [1, 2, 5, 10]
total_train_size = len(train_json)

for per in percentage:
   train_size = int(per * total_train_size / 100)
   train_set = sample(train_json, train_size)
   print('Writing training subset with length = ', len(train_set))
   with open('materials_train' + str(per) + '.json', 'w') as f:
       json.dump(train_set, f, indent = 4)