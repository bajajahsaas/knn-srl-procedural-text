#!/bin/bash
#
#SBATCH --job-name=testing
#SBATCH --output=logstest/test_%j.txt  # output file
#SBATCH -e logstest/test_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#
#SBATCH --ntasks=1

# Exper_train (percen = 5)
python -u test.py --generate --copy --valdata val5.pkl --model_path models/copygen5.pt --test_output_path copy_generate
python -u test.py --no-generate --copy --valdata val5.pkl --model_path models/copy5.pt --test_output_path copy
python -u test.py --generate --no-copy --valdata val5.pkl --model_path models/generate5.pt --test_output_path generate

# Exper_train (percen = 10)
python -u test.py --generate --copy --valdata val10.pkl --model_path models/copygen10.pt --test_output_path copy_generate
python -u test.py --no-generate --copy --valdata val10.pkl --model_path models/copy10.pt --test_output_path copy
python -u test.py --generate --no-copy --valdata val10.pkl --model_path models/generate10.pt --test_output_path generate


# Exper_train (percen = 20)
python -u test.py --generate --copy --valdata val20.pkl --model_path models/copygen20.pt --test_output_path copy_generate
python -u test.py --no-generate --copy --valdata val20.pkl --model_path models/copy20.pt --test_output_path copy
python -u test.py --generate --no-copy --valdata val20.pkl --model_path models/generate20.pt --test_output_path generate

# Exper_train (percen = 50)
python -u test.py --generate --copy --valdata val50.pkl --model_path models/copygen50.pt --test_output_path copy_generate
python -u test.py --no-generate --copy --valdata val50.pkl --model_path models/copy50.pt --test_output_path copy
python -u test.py --generate --no-copy --valdata val50.pkl --model_path models/generate50.pt --test_output_path generate

# Exper_train (percen = 100)
python -u test.py --generate --copy --valdata val100.pkl --model_path models/copygen100.pt --test_output_path copy_generate
python -u test.py --no-generate --copy --valdata val100.pkl --model_path models/copy100.pt --test_output_path copy
python -u test.py --generate --no-copy --valdata val100.pkl --model_path models/generate100.pt --test_output_path generate



## on Test split
# Exper_train (percen = 5)
python -u test.py --generate --copy --valdata test5.pkl --model_path models/copygen5.pt --test_output_path copy_generate_test
python -u test.py --no-generate --copy --valdata test5.pkl --model_path models/copy5.pt --test_output_path copy_test
python -u test.py --generate --no-copy --valdata test5.pkl --model_path models/generate5.pt --test_output_path generate_test

# Exper_train (percen = 10)
python -u test.py --generate --copy --valdata test10.pkl --model_path models/copygen10.pt --test_output_path copy_generate_test
python -u test.py --no-generate --copy --valdata test10.pkl --model_path models/copy10.pt --test_output_path copy_test
python -u test.py --generate --no-copy --valdata test10.pkl --model_path models/generate10.pt --test_output_path generate_test


# Exper_train (percen = 20)
python -u test.py --generate --copy --valdata test20.pkl --model_path models/copygen20.pt --test_output_path copy_generate_test
python -u test.py --no-generate --copy --valdata test20.pkl --model_path models/copy20.pt --test_output_path copy_test
python -u test.py --generate --no-copy --valdata test20.pkl --model_path models/generate20.pt --test_output_path generate_test

# Exper_train (percen = 50)
python -u test.py --generate --copy --valdata test50.pkl --model_path models/copygen50.pt --test_output_path copy_generate_test
python -u test.py --no-generate --copy --valdata test50.pkl --model_path models/copy50.pt --test_output_path copy_test
python -u test.py --generate --no-copy --valdata test50.pkl --model_path models/generate50.pt --test_output_path generate_test

# Exper_train (percen = 100)
python -u test.py --generate --copy --valdata test100.pkl --model_path models/copygen100.pt --test_output_path copy_generate_test
python -u test.py --no-generate --copy --valdata test100.pkl --model_path models/copy100.pt --test_output_path copy_test
python -u test.py --generate --no-copy --valdata test100.pkl --model_path models/generate100.pt --test_output_path generate_test
#sleep 1
exit
