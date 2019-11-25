#!/bin/bash
#
#SBATCH --job-name=copygen-no-entity
#SBATCH --output=logscopygen_noentity/copygen_no_entity%j.txt  # output file
#SBATCH -e logscopygen_noentity/copygen_no_entity%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH --partition=m40-long
#
#SBATCH --ntasks=1

python -u train.py --model_path models/copygen_no_entity.pt --plot_path logscopygen_noentity  --gpu --no-entity
#sleep 1
exit
