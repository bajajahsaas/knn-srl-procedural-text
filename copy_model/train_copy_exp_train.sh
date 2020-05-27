#!/bin/bash
#
#SBATCH --job-name=copy_exp_t
#SBATCH --output=logscopy_exp_train/copy_%j.txt  # output file
#SBATCH -e logscopy_exp_train/copy_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx-long # Partition to submit to
#SBATCH --mem=40GB
#
#SBATCH --ntasks=1

python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/run-2/train1.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/run-2/val1.pkl --model_path models/copy1.pt --plot_path logscopy_exp_train/1 --no-generate --gpu
python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/run-2/train2.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/run-2/val2.pkl --model_path models/copy2.pt --plot_path logscopy_exp_train/2 --no-generate --gpu

# python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/run-1/train5.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/run-1/val5.pkl --model_path models/copy5.pt --plot_path logscopy_exp_train/5 --no-generate --gpu
# python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/run-1/train10.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/run-1/val10.pkl --model_path models/copy10.pt --plot_path logscopy_exp_train/10 --no-generate --gpu
# python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/run-1/train20.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/run-1/val20.pkl --model_path models/copy20.pt --plot_path logscopy_exp_train/20 --no-generate --gpu
# python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/run-1/train50.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/run-1/val50.pkl --model_path models/copy50.pt --plot_path logscopy_exp_train/50 --no-generate --gpu
# python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/run-1/train100.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/run-1/val100.pkl --model_path models/copy100.pt --plot_path logscopy_exp_train/100 --no-generate --gpu

#sleep 1
exit
