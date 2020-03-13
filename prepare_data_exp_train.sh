#!/bin/bash
#
#SBATCH --job-name=copy
#SBATCH --output=logsprepcontext/copy_%j.txt  # output file
#SBATCH -e logsprepcontext/copy_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#
#SBATCH --ntasks=1



python preprocess_wetlabs_exp_train.py

python build_index_from_json_exp_train.py scibert

python copy_model/prepare_data.py wetlabs_train1.json scibert train_embeddings1.pkl
python copy_model/prepare_data.py wetlabs_train2.json scibert train_embeddings2.pkl
python copy_model/prepare_data.py wetlabs_train5.json scibert train_embeddings5.pkl
#python copy_model/prepare_data.py wetlabs_train10.json scibert train_embeddings10.pkl
#python copy_model/prepare_data.py wetlabs_train20.json scibert train_embeddings20.pkl
#python copy_model/prepare_data.py wetlabs_train50.json scibert train_embeddings50.pkl
#python copy_model/prepare_data.py wetlabs_train100.json scibert train_embeddings100.pkl

#python copy_model/prepare_data.py wetlabs_val.json scibert val_embeddings.pkl
#python copy_model/prepare_data.py wetlabs_test.json scibert test_embeddings.pkl

python copy_model/prepare_context.py original1.annoy original_bert.pkl train_embeddings1.pkl train_embeddings1.pkl train1.pkl scibert 4 5
python copy_model/prepare_context.py original1.annoy original_bert.pkl train_embeddings1.pkl test_embeddings.pkl test1.pkl scibert 4 5
python copy_model/prepare_context.py original1.annoy original_bert.pkl train_embeddings1.pkl val_embeddings.pkl val1.pkl scibert 4 5

python copy_model/prepare_context.py original2.annoy original_bert.pkl train_embeddings2.pkl train_embeddings2.pkl train2.pkl scibert 4 5
python copy_model/prepare_context.py original2.annoy original_bert.pkl train_embeddings2.pkl test_embeddings.pkl test2.pkl scibert 4 5
python copy_model/prepare_context.py original2.annoy original_bert.pkl train_embeddings2.pkl val_embeddings.pkl val2.pkl scibert 4 5

python copy_model/prepare_context.py original5.annoy original_bert.pkl train_embeddings5.pkl train_embeddings5.pkl train5.pkl scibert 4 5
python copy_model/prepare_context.py original5.annoy original_bert.pkl train_embeddings5.pkl test_embeddings.pkl test5.pkl scibert 4 5
python copy_model/prepare_context.py original5.annoy original_bert.pkl train_embeddings5.pkl val_embeddings.pkl val5.pkl scibert 4 5

#python copy_model/prepare_context.py original10.annoy original_bert.pkl train_embeddings10.pkl train_embeddings10.pkl train10.pkl scibert 4 5
#python copy_model/prepare_context.py original10.annoy original_bert.pkl train_embeddings10.pkl test_embeddings.pkl test10.pkl scibert 4 5
#python copy_model/prepare_context.py original10.annoy original_bert.pkl train_embeddings10.pkl val_embeddings.pkl val10.pkl scibert 4 5
#
#python copy_model/prepare_context.py original20.annoy original_bert.pkl train_embeddings20.pkl train_embeddings20.pkl train20.pkl scibert 4 5
#python copy_model/prepare_context.py original20.annoy original_bert.pkl train_embeddings20.pkl test_embeddings.pkl test20.pkl scibert 4 5
#python copy_model/prepare_context.py original20.annoy original_bert.pkl train_embeddings20.pkl val_embeddings.pkl val20.pkl scibert 4 5
#
#python copy_model/prepare_context.py original50.annoy original_bert.pkl train_embeddings50.pkl train_embeddings50.pkl train50.pkl scibert 4 5
#python copy_model/prepare_context.py original50.annoy original_bert.pkl train_embeddings50.pkl test_embeddings.pkl test50.pkl scibert 4 5
#python copy_model/prepare_context.py original50.annoy original_bert.pkl train_embeddings50.pkl val_embeddings.pkl val50.pkl scibert 4 5
#
#python copy_model/prepare_context.py original100.annoy original_bert.pkl train_embeddings100.pkl train_embeddings100.pkl train100.pkl scibert 4 5
#python copy_model/prepare_context.py original100.annoy original_bert.pkl train_embeddings100.pkl test_embeddings.pkl test100.pkl scibert 4 5
#python copy_model/prepare_context.py original100.annoy original_bert.pkl train_embeddings100.pkl val_embeddings.pkl val100.pkl scibert 4 5

exit
