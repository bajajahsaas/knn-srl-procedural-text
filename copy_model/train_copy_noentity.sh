#!/bin/bash
#
#SBATCH --job-name=copy-no-entity
#SBATCH --output=logscopy_noentity/copy_no_entity%j.txt  # output file
#SBATCH -e logscopy_noentity/copy_no_entity%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#
#SBATCH --ntasks=1

python -u train.py --model_path models/copy_no_entity.pt --plot_path logscopy_noentity --no-generate --gpu --no-entity
#sleep 1
exit
