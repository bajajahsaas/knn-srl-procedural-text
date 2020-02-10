#!/bin/bash
#
#SBATCH --job-name=generate
#SBATCH --output=logsgen_exp_train/copy_%j.txt  # output file
#SBATCH -e logsgen_exp_train/copy_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --mem=10000
#
#SBATCH --ntasks=1

python -u train.py --traindata train5.pkl --valdata val5.pkl --model_path models/generate5.pt --plot_path logsgen_exp_train/5 --no-copy --generate --gpu --classes 16 --num_entities 21 --num_buckets 11 --batch_size 8
python -u train.py --traindata train10.pkl --valdata val10.pkl --model_path models/generate10.pt --plot_path logsgen_exp_train/10 --no-copy --generate --gpu --classes 16 --num_entities 21 --num_buckets 11 --batch_size 8
python -u train.py --traindata train20.pkl --valdata val20.pkl --model_path models/generate20.pt --plot_path logsgen_exp_train/20 --no-copy --generate --gpu --classes 16 --num_entities 21 --num_buckets 11 --batch_size 8
python -u train.py --traindata train50.pkl --valdata val50.pkl --model_path models/generate50.pt --plot_path logsgen_exp_train/50 --no-copy --generate --gpu --classes 16 --num_entities 21 --num_buckets 11 --batch_size 8
python -u train.py --traindata train100.pkl --valdata val100.pkl --model_path models/generate100.pt --plot_path logsgen_exp_train/100 --no-copy --generate --gpu --classes 16 --num_entities 21 --num_buckets 11 --batch_size 8
#sleep 1
exit
