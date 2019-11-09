#!/bin/bash
#
#SBATCH --job-name=copy_gen_biobert
#SBATCH --output=logs/copy_gen_biobert_%j.txt  # output file
#SBATCH -e logs/copy_gen_biobert_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#
#SBATCH --ntasks=1

python -u train.py --model_path models/copy_generate_biobert.pt --gpu --traindata train_biobert.pkl --valdata val_biobert.pkl 
#sleep 1
exit
