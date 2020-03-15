#!/bin/bash
#
#SBATCH --job-name=generate
#SBATCH --output=logsgen/generate_%j.txt  # output file
#SBATCH -e logsgen/generate_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:2
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --mem=10000
#
#SBATCH --ntasks=1

python -u train_finetune.py --model_path models/generate.pt --plot_path logsgen --no-copy --gpu --traindata train_ft.pkl --valdata val_ft.pkl --lr 0.00001
#sleep 1
exit
