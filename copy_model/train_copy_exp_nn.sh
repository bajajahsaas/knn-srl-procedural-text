#!/bin/bash
#
#SBATCH --job-name=copy
#SBATCH --output=logscopy_exp_nn/copy_%j.txt  # output file
#SBATCH -e logscopy_exp_nn/copy_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --mem=50GB
#
#SBATCH --ntasks=1

python -u train.py --traindata train_NN_10.pkl --valdata val_NN_10.pkl --model_path models/copy_nn10.pt --plot_path logscopy_exp_nn/10 --no-generate --copy --gpu
python -u train.py --traindata train_NN_20.pkl --valdata val_NN_20.pkl --model_path models/copy_nn20.pt --plot_path logscopy_exp_nn/20 --no-generate --copy --gpu
#sleep 1
exit
