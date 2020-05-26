#!/bin/bash
#
#SBATCH --job-name=preparedata
#SBATCH --output=logsprepcontext/copy_%j.txt  # output file
#SBATCH -e logsprepcontext/copy_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=1080ti-short # Partition to submit to
#SBATCH --mem=40GB
#
#SBATCH --ntasks=1

# mnt=/mnt/nfs/work1/mccallum/dswarupogguv

datapath=$mnt/preprocessed_wetlab_data
final=$datapath/final

mkdir -p $final

python preprocess_wetlabs_exp_train.py

python build_index_from_json_wetlabs_exp_train.py scibert

cp wetlab_config/* .

# percentages=(1 2 5)
# ks=(1 2 5)

percentages=(1 2 5 10 20 50 100)
ks=(1 2 5 10 15 20 30 40 50)


for per in ${percentages[@]}; do
    python copy_model/prepare_data.py wetlabs_train${per}.json scibert ${datapath}/train_embeddings${per}.pkl
    python copy_model/prepare_data.py wetlabs_val${per}.json scibert ${datapath}/val_embeddings${per}.pkl
done

python copy_model/prepare_data.py wetlabs_test.json scibert ${datapath}/test_embeddings.pkl

for per in ${percentages[@]}; do
	# always train with k = 5
	echo "Here"
        python copy_model/prepare_context.py original${per}.annoy original_bert.pkl ${datapath}/train_embeddings${per}.pkl ${datapath}/train_embeddings${per}.pkl ${final}/train${per}.pkl scibert 4 5
        for k in ${ks[@]}; do
            python copy_model/prepare_context.py original${per}.annoy original_bert.pkl ${datapath}/train_embeddings${per}.pkl ${datapath}/test_embeddings.pkl ${final}/test${per}.pkl scibert 4 ${k}
            python copy_model/prepare_context.py original${per}.annoy original_bert.pkl ${datapath}/train_embeddings${per}.pkl ${datapath}/val_embeddings${per}.pkl ${final}/val${per}.pkl scibert 4 ${k}
	done
done


exit
