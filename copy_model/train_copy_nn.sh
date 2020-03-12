#!/bin/bash
#
#SBATCH --job-name=copy
#SBATCH --output=logscopy/copy_%j.txt  # output file
#SBATCH -e logscopy/copy_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=2080ti-long # Partition to submit to
#SBATCH --mem=40GB
#
#SBATCH --ntasks=1

python -u train.py --model_path models/copy_nn_10.pt --plot_path logscopy/nn_10 --no-generate --traindata train_NN_10.pkl --valdata val_NN_10.pkl --batch_size 1
python -u train.py --model_path models/copy_nn_20.pt --plot_path logscopy/nn_20 --no-generate --traindata train_NN_20.pkl --valdata val_NN_20.pkl --batch_size 1
#sleep 1
exit
