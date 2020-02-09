#!/bin/bash
#
#SBATCH --job-name=generate
#SBATCH --output=logsgen/generate_%j.txt  # output file
#SBATCH -e logsgen/generate_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --mem=10000
#
#SBATCH --ntasks=1

python -u train.py --model_path models/generate.pt --plot_path logsgen --no-copy --gpu
#sleep 1
exit
