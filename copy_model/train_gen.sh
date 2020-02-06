#!/bin/bash
#
#SBATCH --job-name=generate
#SBATCH --output=logsgen/generate_%j.txt  # output file
#SBATCH -e logsgen/generate_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --mem=100000
#SBATCH --partition=titanx-long # Partition to submit to
#SBATCH --ntasks=1

python -u train.py --model_path models/generate.pt --plot_path logsgen --no-copy --gpu --classes 16
#sleep 1
exit
