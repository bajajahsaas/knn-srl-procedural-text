#!/bin/bash
#
#SBATCH --job-name=testing
#SBATCH --output=logstest/test_%j.txt  # output file
#SBATCH -e logstest/test_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --partition=1080ti-short
#
#SBATCH --ntasks=1

# Exper_train (percen = 1)
# python -u test.py --generate --copy --valdata val1.pkl --model_path models/copygen1.pt --test_output_path copy_generate
for per in 1 2 5 10 20 50 100; do
    python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/val${per}_k40.pkl --model_path /mnt/nfs/work1/mccallum/abajaj/akbc/models/wlp-all-feats-bertretr/prototype/exp_train/May28/wtlabs_copy${per}.pt --test_output_path copy --gpu
    python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/test${per}_k40.pkl --model_path /mnt/nfs/work1/mccallum/abajaj/akbc/models/wlp-all-feats-bertretr/prototype/exp_train/May28/wtlabs_copy${per}.pt --test_output_path copy --gpu
done
# python -u test.py --generate --no-copy --valdata val1.pkl --model_path models/generate1.pt --test_output_path generate

# # Exper_train (percen = 2)
# # python -u test.py --generate --copy --valdata val2.pkl --model_path models/copygen2.pt --test_output_path copy_generate
# python -u test.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/val2_k40.pkl --model_path models/copy2.pt --test_output_path copy --gpu
# # python -u test.py --generate --no-copy --valdata val2.pkl --model_path models/generate2.pt --test_output_path generate
# 
# # Exper_train (percen = 5)
# # python -u test.py --generate --copy --valdata val5.pkl --model_path models/copygen5.pt --test_output_path copy_generate
# python -u test.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/val5_k40.pkl --model_path models/copy5.pt --test_output_path copy --gpu
# # python -u test.py --generate --no-copy --valdata val5.pkl --model_path models/generate5.pt --test_output_path generate
# 
# ## Exper_train (percen = 10)
# #python -u test.py --generate --copy --valdata val10.pkl --model_path models/copygen10.pt --test_output_path copy_generate
# python -u test.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/val10_k40.pkl --model_path models/copy10.pt --test_output_path copy --gpu
# # python -u test.py --generate --no-copy --valdata val10.pkl --model_path models/generate10.pt --test_output_path generate
# #
# ## Exper_train (percen = 20)
# #python -u test.py --generate --copy --valdata val20.pkl --model_path models/copygen20.pt --test_output_path copy_generate
# python -u test.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/val20_k40.pkl --model_path models/copy20.pt --test_output_path copy --gpu
# #python -u test.py --generate --no-copy --valdata val20.pkl --model_path models/generate20.pt --test_output_path generate
# #
# ## Exper_train (percen = 50)
# #python -u test.py --generate --copy --valdata val50.pkl --model_path models/copygen50.pt --test_output_path copy_generate
# python -u test.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/val50_k40.pkl --model_path models/copy50.pt --test_output_path copy --gpu
# #python -u test.py --generate --no-copy --valdata val50.pkl --model_path models/generate50.pt --test_output_path generate
# 
# #python -u test.py --generate --copy --valdata val50.pkl --model_path models/copygen50.pt --test_output_path copy_generate
# python -u test.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/val100_k40.pkl --model_path models/copy100.pt --test_output_path copy --gpu
# #python -u test.py --generate --no-copy --valdata val50.pkl --model_path models/generate50.pt --test_output_path generate



# # K = 40
# # Exper_train (percen = 1)
# # python -u test.py --generate --copy --valdata val1_k40.pkl --model_path models/copygen1.pt --test_output_path copy_generate
# python -u test.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/test1_k40.pkl --model_path models/copy1.pt --test_output_path copy --gpu
# # python -u test.py --generate --no-copy --valdata val1_k40.pkl --model_path models/generate1.pt --test_output_path generate
# 
# # # Exper_train (percen = 2)
# # # python -u test.py --generate --copy --valdata val2_k40.pkl --model_path models/copygen2.pt --test_output_path copy_generate
# python -u test.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/test2_k40.pkl --model_path models/copy2.pt --test_output_path copy --gpu
# # python -u test.py --generate --no-copy --valdata val2_k40.pkl --model_path models/generate2.pt --test_output_path generate
# 
# # # Exper_train (percen = 5)
# # # python -u test.py --generate --copy --valdata val5_k40.pkl --model_path models/copygen5.pt --test_output_path copy_generate
# python -u test.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/test5_k40.pkl --model_path models/copy5.pt --test_output_path copy --gpu
# # python -u test.py --generate --no-copy --valdata val5_k40.pkl --model_path models/generate5.pt --test_output_path generate
# 
# # ## Exper_train (percen = 10)
# # #python -u test.py --generate --copy --valdata val10_k40.pkl --model_path models/copygen10.pt --test_output_path copy_generate
# python -u test.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/test10_k40.pkl --model_path models/copy10.pt --test_output_path copy --gpu
# # python -u test.py --generate --no-copy --valdata val10_k40.pkl --model_path models/generate10.pt --test_output_path generate
# 
# # # Exper_train (percen = 2)
# # # python -u test.py --generate --copy --valdata val2_k40.pkl --model_path models/copygen2.pt --test_output_path copy_generate
# python -u test.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/test20_k40.pkl --model_path models/copy20.pt --test_output_path copy --gpu
# # python -u test.py --generate --no-copy --valdata val2_k40.pkl --model_path models/generate2.pt --test_output_path generate
# 
# # # Exper_train (percen = 5)
# # # python -u test.py --generate --copy --valdata val5_k40.pkl --model_path models/copygen5.pt --test_output_path copy_generate
# python -u test.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/test50_k40.pkl --model_path models/copy50.pt --test_output_path copy --gpu
# # python -u test.py --generate --no-copy --valdata val5_k40.pkl --model_path models/generate5.pt --test_output_path generate
# 
# # ## Exper_train (percen = 10)
# # #python -u test.py --generate --copy --valdata val10_k40.pkl --model_path models/copygen10.pt --test_output_path copy_generate
# python -u test.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/scibert-ret-exp-train/exp-k-test/new/test100_k40.pkl --model_path models/copy100.pt --test_output_path copy --gpu
# # python -u test.py --generate --no-copy --valdata val10_k40.pkl --model_path models/generate10.pt --test_output_path generate
# 
#sleep 1
exit
