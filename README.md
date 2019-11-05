# Learning-to-understand-scientific-experiments

## Crawling
```
python crawl.py
```


## Nearest neighbors in tf-idf space
```python 
python build_index.py
python test_nn.py replaced_tfidf.pkl replaced.annoy replaced_sentences.txt original_sentences.txt
```
## Test upper bound performance
```python 
python test_upperbound.py
python uppperbound_recall.py
```

## Retrieve and Edit Model

### To Preprocess protocols data, split into train and test, writing into json 
```python 
python preprocess_wetlabs.py
```

### Using wetlabs_train.json to build annoy files and replaced_$model$.pkl file of vectorizer for KNN retrieval
```python 
python build_index_from_json.py
```

### Generate embeddings of training data using BERT (or bioBERT). Writes into train_embedding_data.pkl
```python 
python copy_model/prepare_data.py wetlabs_train.json
```
### Using pre-generated annoy, vectorizer, BERT embeddings to pre-compute nearest neighbors (for edit model)
```python 
python copy_model/prepare_context.py replaced.annoy replaced_tfidf.pkl train_embedding_data.pkl
```
