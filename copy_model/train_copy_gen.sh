#!/bin/bash
#
#SBATCH --job-name=copy_gen
#SBATCH --output=logs/copy_gen_%j.txt  # output file
#SBATCH -e logs/copy_gen_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#
#SBATCH --ntasks=1

python -u train.py --model_path models/copy_generate.pt --gpu
#sleep 1
exit
