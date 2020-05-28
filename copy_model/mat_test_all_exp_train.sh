#!/bin/bash
#
#SBATCH --job-name=testing
#SBATCH --output=logstest/test_%j.txt  # output file
#SBATCH -e logstest/test_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long
#SBATCH --mem=50G
#
#SBATCH --ntasks=1

## on Test split
# Exper_train (K = 1)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newtestk1.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/test1_k40.pkl --model_path models/copy1.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newtestk1.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2

# Exper_train (K = 2)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newtestk2.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/test2_k40.pkl --model_path models/copy2.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newtestk2.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2

# Exper_train (K = 5)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newtestk5.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/test5_k40.pkl --model_path models/copy5.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newtestk5.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2

# Exper_train (K = 10)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newtestk10.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/test10_k40.pkl --model_path models/copy10.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newtestk10.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2

# Exper_train (K = 15)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newtestk15.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/test20_k40.pkl --model_path models/copy20.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newtestk15.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2

# Exper_train (K = 20)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newtestk20.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/test50_k40.pkl --model_path models/copy50.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newtestk20.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2

# Exper_train (K = 30)
#python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newtestk30.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/test100_k40.pkl --model_path models/copy100.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 1
#python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newtestk30.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2


## on Val split
# # Exper_train (K = 1)
# #python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/valk1.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/val1_k40.pkl --model_path models/copy1.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
# #python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newvalk1.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2

# # Exper_train (K = 2)
# #python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newvalk2.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/val2_k40.pkl --model_path models/copy2.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
# #python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newvalk2.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2

# # Exper_train (K = 5)
# #python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newvalk5.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/val5_k40.pkl --model_path models/copy5.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
# #python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newvalk5.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2

# # Exper_train (K = 10)
# #python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newvalk10.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/val10_k40.pkl --model_path models/copy10.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
# #python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newvalk10.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2

# # Exper_train (K = 15)
# #python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newvalk15.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/val20_k40.pkl --model_path models/copy20.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
# #python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newvalk15.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2

# # Exper_train (K = 20)
# #python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newvalk20.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/val50_k40.pkl --model_path models/copy50.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
# #python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newvalk20.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2

# # Exper_train (K = 30)
# #python -u test_k.py --generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newvalk30.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
python -u test_k.py --no-generate --copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/new/val100_k40.pkl --model_path models/copy100.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 1
# #python -u test_k.py --generate --no-copy --valdata /mnt/nfs/work1/mccallum/abajaj/akbc/data/materials-scibert-retr-exp-train/newvalk30.pkl --model_path models/generate.pt --test_output_path generate_test --classes 16 --num_entities 21 --num_buckets 11 --gpu --batch_size 2
