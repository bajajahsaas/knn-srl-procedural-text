#!/bin/bash
#
#SBATCH --job-name=generate-no-entity
#SBATCH --output=logsgen_noentity/generate_no_entity%j.txt  # output file
#SBATCH -e logsgen_noentity/generate_no_entity%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#
#SBATCH --ntasks=1

python -u train.py --model_path models/generate_no_entity.pt --plot_path logsgen_noentity --no-copy --gpu --no-entity
#sleep 1
exit
