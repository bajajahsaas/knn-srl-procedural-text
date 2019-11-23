#!/bin/bash
#
#SBATCH --job-name=copy
#SBATCH --output=logscopy_exp_k/copy_%j.txt  # output file
#SBATCH -e logscopy_exp_k/copy_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --mem=10000
#
#SBATCH --ntasks=1

python -u train.py --traindata train100_K10.pkl --valdata val100_K10.pkl --model_path models/copy100_K10.pt --plot_path logscopy_exp_k/10 --no-generate --gpu
python -u train.py --traindata train100_K20.pkl --valdata val100_K20.pkl --model_path models/copy100_K20.pt --plot_path logscopy_exp_k/20 --no-generate --gpu

#sleep 1
exit
