#!/bin/bash
#
#SBATCH --job-name=prepare_context
#SBATCH --output=logsprepcontext_exp_k/copy_%j.txt  # output file
#SBATCH -e logsprepcontext_exp_k/copy_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#
#SBATCH --ntasks=1


python -u copy_model/prepare_context.py replaced100.annoy replaced_tfidf100.pkl train_embeddings100.pkl train_embeddings100.pkl train100.pkl
python -u copy_model/prepare_context.py replaced100.annoy replaced_tfidf100.pkl train_embeddings100.pkl test_embeddings.pkl test100.pkl
python -u copy_model/prepare_context.py replaced100.annoy replaced_tfidf100.pkl train_embeddings100.pkl val_embeddings.pkl val100.pkl

exit
