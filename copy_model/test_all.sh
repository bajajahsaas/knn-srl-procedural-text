python test_joint.py --generate --no-copy  --model_path models/generate.pt --test_output_path generate
python test_joint.py --no-generate --copy --model_path models/copy.pt --test_output_path copy
python test_joint.py --generate --copy  --model_path models/copy_generate.pt --test_output_path copy_generate
