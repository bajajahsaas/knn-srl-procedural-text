#!/bin/bash
#
#SBATCH --job-name=generate
#SBATCH --output=logs/generate_%j.txt  # output file
#SBATCH -e logs/generate_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#
#SBATCH --ntasks=1

python -u train.py --model_path models/generate.pt --no-copy --gpu
#sleep 1
exit
