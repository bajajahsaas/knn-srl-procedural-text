#!/bin/bash
#
#SBATCH --job-name=generate
#SBATCH --output=logsgen_exp_train/copy_%j.txt  # output file
#SBATCH -e logsgen_exp_train/copy_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx-long # Partition to submit to
#SBATCH --mem=40GB
#
#SBATCH --ntasks=1

python -u train.py --traindata train1.pkl --valdata val1.pkl --model_path models/generate1.pt --plot_path logsgen_exp_train/1 --no-copy --gpu
python -u train.py --traindata train2.pkl --valdata val2.pkl --model_path models/generate2.pt --plot_path logsgen_exp_train/2 --no-copy --gpu
python -u train.py --traindata train5.pkl --valdata val5.pkl --model_path models/generate5.pt --plot_path logsgen_exp_train/5 --no-copy --gpu
python -u train.py --traindata train10.pkl --valdata val10.pkl --model_path models/generate10.pt --plot_path logsgen_exp_train/10 --no-copy --gpu
#python -u train.py --traindata train20.pkl --valdata val20.pkl --model_path models/generate20.pt --plot_path logsgen_exp_train/20 --no-copy --gpu
#python -u train.py --traindata train50.pkl --valdata val50.pkl --model_path models/generate50.pt --plot_path logsgen_exp_train/50 --no-copy --gpu
#sleep 1
exit
