#!/bin/bash
#
#SBATCH --job-name=data_prep
#SBATCH --output=logsprepdata/test_%j.txt  # output file
#SBATCH -e logsprepdata/test_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
SBATCH --partition=m40-short # Partition to submit to
#SBATCH --mem=50GB
#
#SBATCH --ntasks=1

#python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl test_embeddings.pkl testk1.pkl scibert 4 1
#python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl test_embeddings.pkl testk2.pkl scibert 4 2
#python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl test_embeddings.pkl testk5.pkl scibert 4 5
#python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl test_embeddings.pkl testk10.pkl scibert 4 10
#python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl test_embeddings.pkl testk15.pkl scibert 4 15
#python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl test_embeddings.pkl testk20.pkl scibert 4 20
#python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl test_embeddings.pkl testk30.pkl scibert 4 30
#python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl test_embeddings.pkl testk40.pkl scibert 4 40
#python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl test_embeddings.pkl testk50.pkl scibert 4 50

# python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl val_embeddings.pkl valk1.pkl scibert 4 1
# python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl val_embeddings.pkl valk2.pkl scibert 4 2
# python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl val_embeddings.pkl valk5.pkl scibert 4 5
# python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl val_embeddings.pkl valk10.pkl scibert 4 10
# python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl val_embeddings.pkl valk15.pkl scibert 4 15
# python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl val_embeddings.pkl valk20.pkl scibert 4 20
# python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl val_embeddings.pkl valk30.pkl scibert 4 30
python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl val_embeddings.pkl valk40.pkl scibert 4 40
# python copy_model/prepare_context_k.py original.annoy original_bert.pkl train_embeddings.pkl val_embeddings.pkl valk50.pkl scibert 4 50