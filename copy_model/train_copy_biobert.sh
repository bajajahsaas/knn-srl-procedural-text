#!/bin/bash
#
#SBATCH --job-name=copy_biobert
#SBATCH --output=logs/copy_biobert_%j.txt  # output file
#SBATCH -e logs/copy_biobert_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#
#SBATCH --ntasks=1

python -u train.py --model_path models/copy_biobert.pt --gpu --traindata train_biobert.pkl --valdata val_biobert.pkl --no-generate
#sleep 1
exit
