import os
from transformers import convert_bert_original_tf_checkpoint_to_pytorch, BertModel, BertTokenizer

# BERT_BASE_DIR='../../biobert_v1.1_pubmed/'
# file = BERT_BASE_DIR + 'pytorch_model.bin'
#
# if os.path.exists(file):
#     print("PyTorch version of bioBERT found")
# else:
#     print('Convert tf checkpoint to pyTorch')
#     convert_bert_original_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(BERT_BASE_DIR + 'model.ckpt-1000000', BERT_BASE_DIR + 'bert_config.json' , BERT_BASE_DIR + 'pytorch_model.bin')

BIOBERT_DIR = '../../biobert' # Rename bert_config.json to config.json


def getbiobertmodel():
    tokenizer = BertTokenizer.from_pretrained(BIOBERT_DIR)
    model = BertModel.from_pretrained(BIOBERT_DIR)
    return tokenizer, model


