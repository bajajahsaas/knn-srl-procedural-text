#!/bin/bash
#
#SBATCH --job-name=copy_max
#SBATCH --output=logscopy/copy_%j.txt  # output file
#SBATCH -e logscopy/copy_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --mem=40000
#
#SBATCH --ntasks=1
mnt=/mnt/nfs/work1/mccallum/dswarupogguv
python -u train.py --model_path models/copy_max_pooling.pt --plot_path logscopy --no-generate --copy --traindata $mnt/exact/data/train.pkl --valdata $mnt/exact/data/val.pkl --batch_size 16 --gpu --classes 16 --num_entities 21 --num_buckets 11

#sleep 1
exit
