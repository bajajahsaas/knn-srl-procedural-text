# python build_index_from_json.py scibert
# 
# python copy_model/prepare_data.py wetlabs_train.json scibert train_embeddings.pkl
# python copy_model/prepare_data.py wetlabs_val.json scibert val_embeddings.pkl
# python copy_model/prepare_data.py wetlabs_test.json scibert test_embeddings.pkl

python copy_model/prepare_context_finetune.py original.annoy original_bert.pkl train_embeddings.pkl train_embeddings.pkl train_ft.pkl scibert 4
python copy_model/prepare_context_finetune.py original.annoy original_bert.pkl train_embeddings.pkl test_embeddings.pkl test_ft.pkl scibert 4
python copy_model/prepare_context_finetune.py original.annoy original_bert.pkl train_embeddings.pkl val_embeddings.pkl val_ft.pkl scibert 4
