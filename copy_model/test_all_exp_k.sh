#!/bin/bash
#
#SBATCH --job-name=testing
#SBATCH --output=logstest/test_%j.txt  # output file
#SBATCH -e logstest/test_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#
#SBATCH --ntasks=1

# Exper_k (K = 10)
python -u test.py --generate --copy --valdata val100_K10.pkl --model_path models/copygen100_K10.pt --test_output_path copy_generate
python -u test.py --no-generate --copy --valdata val100_K10.pkl --model_path models/copy100_K10.pt --test_output_path copy
python -u test.py --generate --no-copy --valdata val100_K10.pkl --model_path models/generate100_K10.pt --test_output_path generate

# Exper_k (K = 20)
python -u test.py --generate --copy --valdata val100_K20.pkl --model_path models/copygen100_K20.pt --test_output_path copy_generate
python -u test.py --no-generate --copy --valdata val100_K20.pkl --model_path models/copy100_K20.pt --test_output_path copy
python -u test.py --generate --no-copy --valdata val100_K20.pkl --model_path models/generate100_K20.pt --test_output_path generate

#sleep 1
exit
