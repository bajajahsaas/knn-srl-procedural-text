#!/bin/bash
#
#SBATCH --job-name=copy_no_label
#SBATCH --output=logscopy/copy_%j.txt  # output file
#SBATCH -e logscopy/copy_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --mem=50000
#
#SBATCH --ntasks=1

python -u train.py --model_path models/copy_no_label.pt --plot_path logscopy --no-generate --copy --traindata $mnt/exact/data/train.pkl --valdata $mnt/exact/data/val.pkl --batch_size 16 --attnmethod no_label --gpu
#sleep 1
exit
