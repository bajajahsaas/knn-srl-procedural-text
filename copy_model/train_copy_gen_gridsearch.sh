#!/bin/bash
#
#SBATCH --job-name=copy_gen
#SBATCH --output=logscopygen/copy_gen_%j.txt  # output file
#SBATCH -e logscopygen/copy_gen_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#
#SBATCH --ntasks=1

python -u train.py --model_path models/copy_generate.pt --plot_path logscopygen --downsample 0 --gpu
#sleep 1
exit
