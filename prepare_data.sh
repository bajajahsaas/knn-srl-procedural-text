#!/bin/bash
#
#SBATCH --job-name=prepare_data
#SBATCH --output=logsprepdata/prep_data_%j.txt  # output file
#SBATCH -e logsprepdata/prep_data_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=m80-long # Partition to submit to
#SBATCH --mem=10000
#
#SBATCH --ntasks=1

#python -u preprocess_wetlabs.py

python -u build_index_from_json.py scibert

python -u copy_model/prepare_data.py wetlabs_train5.json scibert train_embeddings5.pkl
python -u copy_model/prepare_data.py wetlabs_train10.json scibert train_embeddings10.pkl
python -u copy_model/prepare_data.py wetlabs_train20.json scibert train_embeddings20.pkl
python -u copy_model/prepare_data.py wetlabs_train50.json scibert train_embeddings50.pkl
python -u copy_model/prepare_data.py wetlabs_train100.json scibert train_embeddings100.pkl
python -u copy_model/prepare_data.py wetlabs_val.json scibert val_embeddings.pkl
python -u copy_model/prepare_data.py wetlabs_test.json scibert test_embeddings.pkl

python -u copy_model/prepare_context.py original100.annoy original_bert.pkl train_embeddings100.pkl train_embeddings100.pkl train100.pkl scibert 8
python -u copy_model/prepare_context.py original100.annoy original_bert.pkl train_embeddings100.pkl test_embeddings.pkl test100.pkl scibert 8
python -u copy_model/prepare_context.py original100.annoy original_bert.pkl train_embeddings100.pkl val_embeddings.pkl val100.pkl scibert 8

python -u copy_model/prepare_context.py original5.annoy original_bert.pkl train_embeddings5.pkl train_embeddings5.pkl train5.pkl scibert 8
python -u copy_model/prepare_context.py original5.annoy original_bert.pkl train_embeddings5.pkl test_embeddings.pkl test5.pkl scibert 8
python -u copy_model/prepare_context.py original5.annoy original_bert.pkl train_embeddings5.pkl val_embeddings.pkl val5.pkl scibert 8

python -u copy_model/prepare_context.py original10.annoy original_bert.pkl train_embeddings10.pkl train_embeddings10.pkl train10.pkl scibert 8
python -u copy_model/prepare_context.py original10.annoy original_bert.pkl train_embeddings10.pkl test_embeddings.pkl test10.pkl scibert 8
python -u copy_model/prepare_context.py original10.annoy original_bert.pkl train_embeddings10.pkl val_embeddings.pkl val10.pkl scibert 8

python -u copy_model/prepare_context.py original20.annoy original_bert.pkl train_embeddings20.pkl train_embeddings20.pkl train20.pkl scibert 8
python -u copy_model/prepare_context.py original20.annoy original_bert.pkl train_embeddings20.pkl test_embeddings.pkl test20.pkl scibert 8
python -u copy_model/prepare_context.py original20.annoy original_bert.pkl train_embeddings20.pkl val_embeddings.pkl val20.pkl scibert 8

python -u copy_model/prepare_context.py original50.annoy original_bert.pkl train_embeddings50.pkl train_embeddings50.pkl train50.pkl scibert 8
python -u copy_model/prepare_context.py original50.annoy original_bert.pkl train_embeddings50.pkl test_embeddings.pkl test50.pkl scibert 8
python -u copy_model/prepare_context.py original50.annoy original_bert.pkl train_embeddings50.pkl val_embeddings.pkl val50.pkl scibert 8

exit