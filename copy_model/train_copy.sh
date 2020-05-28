#!/bin/bash
#
#SBATCH --job-name=copy
#SBATCH --output=logscopy/copy_%j.txt  # output file
#SBATCH -e logscopy/copy_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=1080ti-long # Partition to submit to
#SBATCH --mem=40000
#
#SBATCH --ntasks=1

python -u train.py --model_path models/copy.pt --plot_path logscopy --copy --no-generate --traindata train.pkl --valdata val.pkl --batch_size 16 --gpu
#sleep 1
exit
