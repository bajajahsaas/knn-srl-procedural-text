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
#python -u test_with_copy_prob.py --generate --copy --valdata testk1.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
python -u test_with_copy_prob.py --no-generate --copy --valdata testk1.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata testk1.pkl --model_path models/generate.pt --test_output_path generate_test
#
## Exper_train (K = 2)
#python -u test_with_copy_prob.py --generate --copy --valdata testk2.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
python -u test_with_copy_prob.py --no-generate --copy --valdata testk2.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata testk2.pkl --model_path models/generate.pt --test_output_path generate_test
#
## Exper_train (K = 5)
#python -u test_with_copy_prob.py --generate --copy --valdata testk5.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
python -u test_with_copy_prob.py --no-generate --copy --valdata testk5.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata testk5.pkl --model_path models/generate.pt --test_output_path generate_test
#
## Exper_train (K = 10)
#python -u test_with_copy_prob.py --generate --copy --valdata testk10.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
python -u test_with_copy_prob.py --no-generate --copy --valdata testk10.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata testk10.pkl --model_path models/generate.pt --test_output_path generate_test
#
## Exper_train (K = 15)
#python -u test_with_copy_prob.py --generate --copy --valdata testk15.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
python -u test_with_copy_prob.py --no-generate --copy --valdata testk15.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata testk15.pkl --model_path models/generate.pt --test_output_path generate_test

# Exper_train (K = 20)
#python -u test_with_copy_prob.py --generate --copy --valdata testk20.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
python -u test_with_copy_prob.py --no-generate --copy --valdata testk20.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata testk20.pkl --model_path models/generate.pt --test_output_path generate_test

# Exper_train (K = 30)
#python -u test_with_copy_prob.py --generate --copy --valdata testk30.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
python -u test_with_copy_prob.py --no-generate --copy --valdata testk30.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata testk30.pkl --model_path models/generate.pt --test_output_path generate_test

# Exper_train (K = 40)
#python -u test_with_copy_prob.py --generate --copy --valdata testk40.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
python -u test_with_copy_prob.py --no-generate --copy --valdata testk40.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata testk40.pkl --model_path models/generate.pt --test_output_path generate_test

# Exper_train (K = 50)
#python -u test_with_copy_prob.py --generate --copy --valdata testk50.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
python -u test_with_copy_prob.py --no-generate --copy --valdata testk50.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata testk50.pkl --model_path models/generate.pt --test_output_path generate_test


## on Val split
# Exper_train (K = 1)
#python -u test_with_copy_prob.py --generate --copy --valdata valk1.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
#python -u test_with_copy_prob.py --no-generate --copy --valdata valk1.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata valk1.pkl --model_path models/generate.pt --test_output_path generate_test
#
## Exper_train (K = 2)
#python -u test_with_copy_prob.py --generate --copy --valdata valk2.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
#python -u test_with_copy_prob.py --no-generate --copy --valdata valk2.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata valk2.pkl --model_path models/generate.pt --test_output_path generate_test
#
## Exper_train (K = 5)
#python -u test_with_copy_prob.py --generate --copy --valdata valk5.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
#python -u test_with_copy_prob.py --no-generate --copy --valdata valk5.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata valk5.pkl --model_path models/generate.pt --test_output_path generate_test
#
## Exper_train (K = 10)
#python -u test_with_copy_prob.py --generate --copy --valdata valk10.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
#python -u test_with_copy_prob.py --no-generate --copy --valdata valk10.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata valk10.pkl --model_path models/generate.pt --test_output_path generate_test
#
## Exper_train (K = 15)
#python -u test_with_copy_prob.py --generate --copy --valdata valk15.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
#python -u test_with_copy_prob.py --no-generate --copy --valdata valk15.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata valk15.pkl --model_path models/generate.pt --test_output_path generate_test

# Exper_train (K = 20)
#python -u test_with_copy_prob.py --generate --copy --valdata valk20.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
#python -u test_with_copy_prob.py --no-generate --copy --valdata valk20.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata valk20.pkl --model_path models/generate.pt --test_output_path generate_test

# Exper_train (K = 30)
#python -u test_with_copy_prob.py --generate --copy --valdata valk30.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
#python -u test_with_copy_prob.py --no-generate --copy --valdata valk30.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata valk30.pkl --model_path models/generate.pt --test_output_path generate_test

# Exper_train (K = 40)
#python -u test_with_copy_prob.py --generate --copy --valdata valk40.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
#python -u test_with_copy_prob.py --no-generate --copy --valdata valk40.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata valk40.pkl --model_path models/generate.pt --test_output_path generate_test

# Exper_train (K = 50)
#python -u test_with_copy_prob.py --generate --copy --valdata valk50.pkl --model_path models/copy_generate.pt --test_output_path copy_generate_test
#python -u test_with_copy_prob.py --no-generate --copy --valdata valk50.pkl --model_path models/copy.pt --test_output_path copy_test --gpu
#python -u test_with_copy_prob.py --generate --no-copy --valdata valk50.pkl --model_path models/generate.pt --test_output_path generate_test