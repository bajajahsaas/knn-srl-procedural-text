#!/bin/bash
#
#SBATCH --job-name=ner
#SBATCH --output=logsner/bert_crf_%j.txt  # output file
#SBATCH -e logsner/bert_crf_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --mem=100000
#
#SBATCH --ntasks=1

python -u train_ner.py
exit
