#!/bin/bash
#
#SBATCH --job-name=copy
#SBATCH --output=logscopy/copy_%j.txt  # output file
#SBATCH -e logscopy/copy_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#
#SBATCH --ntasks=1

python -u train.py --model_path models/copy.pt --plot_path logscopy --no-generate --gpu
#sleep 1
exit
