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

python -u train.py --traindata train5.pkl --valdata val5.pkl --model_path models/copygen5.pt --plot_path logscopygen_exp_train/5 --generate --copy --gpu --classes 16 --num_entities 21 --num_buckets 11
python -u train.py --traindata train10.pkl --valdata val10.pkl --model_path models/copygen10.pt --plot_path logscopygen_exp_train/10 --generate --copy --gpu --classes 16 --num_entities 21 --num_buckets 11
python -u train.py --traindata train20.pkl --valdata val20.pkl --model_path models/copygen20.pt --plot_path logscopygen_exp_train/20 --generate --copy --gpu --classes 16 --num_entities 21 --num_buckets 11
python -u train.py --traindata train50.pkl --valdata val50.pkl --model_path models/copygen50.pt --plot_path logscopygen_exp_train/50 --generate --copy --gpu --classes 16 --num_entities 21 --num_buckets 11
python -u train.py --traindata train100.pkl --valdata val100.pkl --model_path models/copygen100.pt --plot_path logscopygen_exp_train/100 --generate --copy --gpu --classes 16 --num_entities 21 --num_buckets 11
#sleep 1
exit
