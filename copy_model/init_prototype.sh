#!/bin/bash
#
#SBATCH --job-name=testing
#SBATCH --output=logsinit/test_%j.txt  # output file
#SBATCH -e logsinit/test_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#
#SBATCH --ntasks=1

python -u init_prototype.py --no-generate --copy --traindata train.pkl --model_path models/copy_best.pt --test_output_path init_prototype --gpu
