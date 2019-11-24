#!/bin/bash
#
#SBATCH --job-name=copy_gen
#SBATCH --output=logscopygen_exp_k/copy_gen_%j.txt  # output file
#SBATCH -e logscopygen_exp_k/copy_gen_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --mem=40G
#
#SBATCH --ntasks=1


#python -u train.py --traindata train100_K10.pkl --valdata val100_K10.pkl --model_path models/copygen100_K10.pt --plot_path logscopygen_exp_k/10 --gpu
python -u train.py --traindata train100_K20.pkl --valdata val100_K20.pkl --model_path models/copygen100_K20.pt --plot_path logscopygen_exp_k/20 --gpu

#sleep 1
exit
