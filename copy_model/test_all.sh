#!/bin/bash
#
#SBATCH --job-name=TEST
#SBATCH --output=logscopy/TEST_%j.txt  # output file
#SBATCH -e logscopy/TEST_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=1080ti-short # Partition to submit to
#SBATCH --mem=40000
# python test_joint.py --generate --no-copy  --model_path models/generate.pt --test_output_path generate
#python test_k_output.py --no-generate --copy --model_path models/copy_exact.pt --test_output_path copy --valdata val_k40.pkl --gpu
python test_k_output.py --no-generate --copy --model_path models/copy_no_label.pt --attnmethod no_label --test_output_path copy_no_label --valdata $mnt/exact/data/val_k40.pkl --gpu
# python test_joint.py --generate --copy  --model_path models/copy_generate.pt --test_output_path copy_generate
