#!/bin/bash
#
#SBATCH --job-name=prepare_data
#SBATCH --output=logsprepdata/prep_data_%j.txt  # output file
#SBATCH -e logsprepdata/prep_data_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --mem=10000
#
#SBATCH --ntasks=1


python -u copy_model/prepare_context.py original10.annoy original_bert.pkl train_embeddings10.pkl train_embeddings10.pkl train10.pkl scibert 4

exit

