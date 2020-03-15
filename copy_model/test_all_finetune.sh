#python test_finetune.py --generate --no-copy  --model_path models/generate.pt --test_output_path generate --valdata val_ft.pkl
python test_finetune.py --no-generate --copy --model_path models/copy.pt --test_output_path copy --valdata val_ft.pkl --gpu
#python test_finetune.py --generate --copy  --model_path models/copy_generate.pt --test_output_path copy_generate --valdata val_ft.pkl
