#!/bin/bash
#
#SBATCH --job-name=copy_gen
#SBATCH --output=logscopygen_exp_train/copy_gen_%j.txt  # output file
#SBATCH -e logscopygen_exp_train/copy_gen_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --mem=40G
#
#SBATCH --ntasks=1

python -u train.py --traindata train5.pkl --valdata val5.pkl --model_path models/copygen5.pt --plot_path logscopygen_exp_train/5  --gpu
#python -u train.py --traindata train10.pkl --valdata val10.pkl --model_path models/copygen10.pt --plot_path logscopygen_exp_train/10 --gpu
#python -u train.py --traindata train20.pkl --valdata val20.pkl --model_path models/copygen20.pt --plot_path logscopygen_exp_train/20 --gpu
#python -u train.py --traindata train50.pkl --valdata val50.pkl --model_path models/copygen50.pt --plot_path logscopygen_exp_train/50 --gpu
#sleep 1
exit
