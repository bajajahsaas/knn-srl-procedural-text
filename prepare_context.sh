#!/bin/bash
#
#SBATCH --job-name=prepare_context
#SBATCH --output=logsprepcontext/copy_%j.txt  # output file
#SBATCH -e logsprepcontext/copy_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#
#SBATCH --ntasks=1

python -u copy_model/prepare_context.py replaced5.annoy replaced_tfidf5.pkl train_embeddings5.pkl train_embeddings5.pkl train5.pkl
python -u copy_model/prepare_context.py replaced5.annoy replaced_tfidf5.pkl train_embeddings5.pkl test_embeddings.pkl test5.pkl
python -u copy_model/prepare_context.py replaced5.annoy replaced_tfidf5.pkl train_embeddings5.pkl val_embeddings.pkl val5.pkl


python -u copy_model/prepare_context.py replaced10.annoy replaced_tfidf10.pkl train_embeddings10.pkl train_embeddings10.pkl train10.pkl
python -u copy_model/prepare_context.py replaced10.annoy replaced_tfidf10.pkl train_embeddings10.pkl test_embeddings.pkl test10.pkl
python -u copy_model/prepare_context.py replaced10.annoy replaced_tfidf10.pkl train_embeddings10.pkl val_embeddings.pkl val10.pkl


python -u copy_model/prepare_context.py replaced20.annoy replaced_tfidf20.pkl train_embeddings20.pkl train_embeddings20.pkl train20.pkl
python -u copy_model/prepare_context.py replaced20.annoy replaced_tfidf20.pkl train_embeddings20.pkl test_embeddings.pkl test20.pkl
python -u copy_model/prepare_context.py replaced20.annoy replaced_tfidf20.pkl train_embeddings20.pkl val_embeddings.pkl val20.pkl


python -u copy_model/prepare_context.py replaced50.annoy replaced_tfidf50.pkl train_embeddings50.pkl train_embeddings50.pkl train50.pkl
python -u copy_model/prepare_context.py replaced50.annoy replaced_tfidf50.pkl train_embeddings50.pkl test_embeddings.pkl test50.pkl
python -u copy_model/prepare_context.py replaced50.annoy replaced_tfidf50.pkl train_embeddings50.pkl val_embeddings.pkl val50.pkl


python -u copy_model/prepare_context.py replaced100.annoy replaced_tfidf100.pkl train_embeddings100.pkl train_embeddings100.pkl train100.pkl
python -u copy_model/prepare_context.py replaced100.annoy replaced_tfidf100.pkl train_embeddings100.pkl test_embeddings.pkl test100.pkl
python -u copy_model/prepare_context.py replaced100.annoy replaced_tfidf100.pkl train_embeddings100.pkl val_embeddings.pkl val100.pkl

exit
