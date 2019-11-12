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

### To Preprocess protocols data. Needs WLP-Dataset 
```python 
python preprocess_wetlabs.py
```
This creates 3 files: wetlabs_train.json, wetlabs_val.json, wetlabs_test.json

### Using wetlabs_train.json to build annoy files and vectorizer files for KNN retrieval
```python 
python build_index_from_json.py
```
This creates files original.annoy, replaced.annoy, original_tfidf.pkl and replaced_tfidf.pkl. Annoy indices allow fast nearest neighbor search, tfidf.pkl store the idf components and the vocabulary which is required to vectorize any test sentence.

### Generate embeddings of Sentences using BERT (or bioBERT).
```python 
python copy_model/prepare_data.py wetlabs_train.json biobert train_embeddings.pkl
python copy_model/prepare_data.py wetlabs_val.json biobert val_embeddings.pkl
python copy_model/prepare_data.py wetlabs_test.json biobert test_embeddings.pkl
```
This produces train_embeddings.pkl, val_embeddings.pkl, test_embeddings.pkl. Use argument (biobert) if want to generate embeddings from bioBERT and (bert) to use basebert. 

### Using pre-generated annoy, vectorizer, BERT embeddings to pre-compute nearest neighbors (for edit model)
```python 
python copy_model/prepare_context.py replaced.annoy replaced_tfidf.pkl train_embeddings.pkl train_embeddings.pkl train.pkl
python copy_model/prepare_context.py replaced.annoy replaced_tfidf.pkl train_embeddings.pkl test_embeddings.pkl test.pkl
python copy_model/prepare_context.py replaced.annoy replaced_tfidf.pkl train_embeddings.pkl val_embeddings.pkl val.pkl
```
This computes nearest neighbors in the training set for each sentence in the train, val and test sets. This writes into train.pkl, val.pkl, test.pkl.

### Training using copy and generate mode
```python
python copy_model/train.py --copy \--generate --traindata PATH/TO/train.pkl --valdata PATH/TO/val.pkl --modelpath OUTPUTDIR/model.pt
```

### Training using copy mode
```python 
python copy_model/train.py --copy --no-generate --traindata PATH/TO/train.pkl --valdata PATH/TO/val.pkl --modelpath OUTPUTDIR/model.pt
```

### Training using generate mode
```python 
python copy_model/train.py --generate --no-copy --traindata PATH/TO/train.pkl --valdata PATH/TO/val.pkl --modelpath OUTPUTDIR/model.pt
```
### Test model
Set generate and copy arguments according to how training was done. For example:
```python 
python copy_model/test.py --generate --no-copy --valdata PATH/TO/val.pkl --modelpath OUTPUTDIR/model.pt
```
