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
This creates files wetlabs_train.json and wetlabs_test.json

### Using wetlabs_train.json to build annoy files and vectorizer files for KNN retrieval
```python 
python build_index_from_json.py
```
This creates files original.annoy, replaced.annoy, original_tfidf.pkl and replaced_tfidf.pkl

### Generate embeddings of training data using BERT (or bioBERT).
```python 
python copy_model/prepare_data.py wetlabs_train.json
```
This writes into train_embedding_data.pkl

### Using pre-generated annoy, vectorizer, BERT embeddings to pre-compute nearest neighbors (for edit model)
```python 
python copy_model/prepare_context.py replaced.annoy replaced_tfidf.pkl train_embedding_data.pkl
```
This writes into dataset.pkl to be used for retrieve-and-edit model

### Split dataset.pkl into train and test
```python 
python python copy_model/split_data.py dataset.pkl
```
This generates train.pkl and test.pkl to be used by retrieve and edit model

### Training using copy and generate mode
```python 
python python copy_model/train.py --copy --output.pt
```

### Training using generate mode
```python 
python python copy_model/train.py --nocopy --output.pt
```
