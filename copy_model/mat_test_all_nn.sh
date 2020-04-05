#!/bin/bash
#
#SBATCH --job-name=testing
#SBATCH --output=logstest/test_%j.txt  # output file
#SBATCH -e logstest/test_%j.err        # File to which STDERR will be written
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#
#SBATCH --ntasks=1

## on Test split
# Exper_train (K = 1)
python -u test_k.py --no-generate --copy --valdata testk1.pkl --model_path models/copy_nn_10.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11
#
## Exper_train (K = 2)
python -u test_k.py --no-generate --copy --valdata testk2.pkl --model_path models/copy_nn_10.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11
#
## Exper_train (K = 5)
python -u test_k.py --no-generate --copy --valdata testk5.pkl --model_path models/copy_nn_10.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11
#
## Exper_train (K = 10)
python -u test_k.py --no-generate --copy --valdata testk10.pkl --model_path models/copy_nn_10.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11
#
## Exper_train (K = 15)
python -u test_k.py --no-generate --copy --valdata testk15.pkl --model_path models/copy_nn_10.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11

# Exper_train (K = 20)
python -u test_k.py --no-generate --copy --valdata testk20.pkl --model_path models/copy_nn_10.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11

# Exper_train (K = 30)
python -u test_k.py --no-generate --copy --valdata testk30.pkl --model_path models/copy_nn_10.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11

# Exper_train (K = 40)
python -u test_k.py --no-generate --copy --valdata testk40.pkl --model_path models/copy_nn_10.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11

# Exper_train (K = 50)
python -u test_k.py --no-generate --copy --valdata testk50.pkl --model_path models/copy_nn_10.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11

#####################

 # Exper_train (K = 1)
 python -u test_k.py --no-generate --copy --valdata testk1.pkl --model_path models/copy_nn_20.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11
 #
 ## Exper_train (K = 2)
 python -u test_k.py --no-generate --copy --valdata testk2.pkl --model_path models/copy_nn_20.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11
 #
 ## Exper_train (K = 5)
 python -u test_k.py --no-generate --copy --valdata testk5.pkl --model_path models/copy_nn_20.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11
 #
 ## Exper_train (K = 10)
 python -u test_k.py --no-generate --copy --valdata testk10.pkl --model_path models/copy_nn_20.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11
 #
 ## Exper_train (K = 15)
 python -u test_k.py --no-generate --copy --valdata testk15.pkl --model_path models/copy_nn_20.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11

 # Exper_train (K = 20)
 python -u test_k.py --no-generate --copy --valdata testk20.pkl --model_path models/copy_nn_20.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11

 # Exper_train (K = 30)
 python -u test_k.py --no-generate --copy --valdata testk30.pkl --model_path models/copy_nn_20.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11

 # Exper_train (K = 40)
 python -u test_k.py --no-generate --copy --valdata testk40.pkl --model_path models/copy_nn_20.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11

 # Exper_train (K = 50)
 python -u test_k.py --no-generate --copy --valdata testk50.pkl --model_path models/copy_nn_20.pt --test_output_path copy_test --classes 16 --num_entities 21 --num_buckets 11