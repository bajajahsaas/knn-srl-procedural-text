#!/bin/bash
#
#SBATCH --job-name=copy
#SBATCH --output=logscopy_exp_train/copy_%j.txt  # output file
#SBATCH -e logscopy_exp_train/copy_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=2080ti-long # Partition to submit to
#SBATCH --mem=40G
#
#SBATCH --ntasks=1

python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/train1.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/val1.pkl --model_path models/matsci_copy1.pt --plot_path logscopy_exp_train/1 --no-generate --copy --gpu --classes 16 --num_entities 21 --num_buckets 11 --batch_size 2
python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/train2.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/val2.pkl --model_path models/matsci_copy2.pt --plot_path logscopy_exp_train/2 --no-generate --copy --gpu --classes 16 --num_entities 21 --num_buckets 11 --batch_size 2
python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/train5.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/val5.pkl --model_path models/matsci_copy5.pt --plot_path logscopy_exp_train/5 --no-generate --copy --gpu --classes 16 --num_entities 21 --num_buckets 11 --batch_size 2
python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/train10.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/val10.pkl --model_path models/matsci_copy10.pt --plot_path logscopy_exp_train/10 --no-generate --copy --gpu --classes 16 --num_entities 21 --num_buckets 11 --batch_size 2
python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/train20.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/val20.pkl --model_path models/matsci_copy20.pt --plot_path logscopy_exp_train/20 --no-generate --copy --gpu --classes 16 --num_entities 21 --num_buckets 11 --batch_size 2
python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/train50.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/val50.pkl --model_path models/matsci_copy50.pt --plot_path logscopy_exp_train/50 --no-generate --copy --gpu --classes 16 --num_entities 21 --num_buckets 11 --batch_size 2
python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/train100.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/val100.pkl --model_path models/matsci_copy100.pt --plot_path logscopy_exp_train/100 --no-generate --copy --gpu --classes 16 --num_entities 21 --num_buckets 11 --batch_size 2
#sleep 1
exit
