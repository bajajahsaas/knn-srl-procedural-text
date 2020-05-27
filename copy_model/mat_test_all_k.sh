#!/bin/bash
#
#SBATCH --job-name=testing
#SBATCH --output=logstest/test_%j.txt  # output file
#SBATCH -e logstest/test_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#
#SBATCH --ntasks=1

## on Test split
# Exper_train (K = 1)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk1.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk1.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk1.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu

# Exper_train (K = 2)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk2.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk2.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk2.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu

# Exper_train (K = 5)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk5.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk5.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk5.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu

# Exper_train (K = 10)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk10.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk10.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk10.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu

# Exper_train (K = 15)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk15.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk15.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk15.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu

# Exper_train (K = 20)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk20.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk20.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk20.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu

# Exper_train (K = 30)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk30.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk30.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk30.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu

# Exper_train (K = 40)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk40.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk40.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk40.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu

# Exper_train (K = 50)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk50.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk50.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/testk50.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu



## on Val split
# Exper_train (K = 1)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk1.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk1.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk1.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu

# Exper_train (K = 2)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk2.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk2.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk2.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu

# Exper_train (K = 5)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk5.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk5.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk5.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu

# Exper_train (K = 10)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk10.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk10.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk10.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu

# Exper_train (K = 15)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk15.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk15.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk15.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu

# Exper_train (K = 20)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk20.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk20.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk20.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu

# Exper_train (K = 30)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk30.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk30.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk30.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu

# Exper_train (K = 40)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk40.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk40.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk40.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu

# Exper_train (K = 50)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk50.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk50.pkl --model_path models/copy.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr/exp_k/valk50.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu