#!/bin/bash
#
#SBATCH --job-name=copy_exp_t
#SBATCH --output=logscopy_exp_train/copy_%j.txt  # output file
#SBATCH -e logscopy_exp_train/copy_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=2080ti-long # Partition to submit to
#SBATCH --mem=40GB
#
#SBATCH --ntasks=1

# python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/train1.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/val1.pkl --model_path models/wtlabs_copy1.pt --plot_path logscopy_exp_train/1 --no-generate --gpu
# python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/train2.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/val2.pkl --model_path models/wtlabs_copy2.pt --plot_path logscopy_exp_train/2 --no-generate --gpu
# python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/train5.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/val5.pkl --model_path models/wtlabs_copy5.pt --plot_path logscopy_exp_train/5 --no-generate --gpu
# python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/train10.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/val10.pkl --model_path models/wtlabs_copy10.pt --plot_path logscopy_exp_train/10 --no-generate --gpu
# python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/train20.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/val20.pkl --model_path models/wtlabs_copy20.pt --plot_path logscopy_exp_train/20 --no-generate --gpu
# python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/train50.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/val50.pkl --model_path models/wtlabs_copy50.pt --plot_path logscopy_exp_train/50 --no-generate --gpu
# python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/train100.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/val100.pkl --model_path models/wtlabs_copy100.pt --plot_path logscopy_exp_train/100 --no-generate --gpu

for per in 0.01 0.03 0.06 0.1 0.3 0.6; do
    python -u train.py --traindata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/small/train${per}.pkl --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/small/val${per}.pkl --model_path models/wtlabs_copy${per}.pt --plot_path logscopy_exp_train/${per} --no-generate --gpu
done


#sleep 1
exit
