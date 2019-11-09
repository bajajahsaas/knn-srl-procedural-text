#!/bin/bash
#
#SBATCH --job-name=gen_biobert
#SBATCH --output=logs/gen_biobert_%j.txt  # output file
#SBATCH -e logs/gen_biobert_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#
#SBATCH --ntasks=1

python -u train.py --model_path models/generate_biobert.pt --gpu --traindata train_biobert.pkl --valdata val_biobert.pkl --no-copy 
#sleep 1
exit
