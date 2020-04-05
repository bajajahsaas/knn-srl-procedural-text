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

#python build_index_from_json.py scibert
#
#python copy_model/prepare_data.py wetlabs_train.json scibert train_embeddings.pkl
#python copy_model/prepare_data.py wetlabs_val.json scibert val_embeddings.pkl
#python copy_model/prepare_data.py wetlabs_test.json scibert test_embeddings.pkl

python copy_model/prepare_context.py original.annoy original_bert.pkl train_embeddings.pkl train_embeddings.pkl train_NN_10.pkl scibert 4 10
python copy_model/prepare_context.py original.annoy original_bert.pkl train_embeddings.pkl val_embeddings.pkl val_NN_10.pkl scibert 4 10

python copy_model/prepare_context.py original.annoy original_bert.pkl train_embeddings.pkl train_embeddings.pkl train_NN_20.pkl scibert 4 20
python copy_model/prepare_context.py original.annoy original_bert.pkl train_embeddings.pkl val_embeddings.pkl val_NN_20.pkl scibert 4 20