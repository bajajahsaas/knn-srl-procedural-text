# python build_index_from_json_materials.py scibert
# 
# python copy_model/prepare_data.py materials_train.json scibert train_embeddings.pkl
# python copy_model/prepare_data.py materials_val.json scibert val_embeddings.pkl
# python copy_model/prepare_data.py materials_test.json scibert test_embeddings.pkl

python copy_model/prepare_context.py original.annoy original_bert.pkl train_embeddings.pkl train_embeddings.pkl train.pkl scibert 8
python copy_model/prepare_context.py original.annoy original_bert.pkl train_embeddings.pkl test_embeddings.pkl test.pkl scibert 8
python copy_model/prepare_context.py original.annoy original_bert.pkl train_embeddings.pkl val_embeddings.pkl val.pkl scibert 8
