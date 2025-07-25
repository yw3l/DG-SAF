python preprocess.py --data_dir data/FB15k237 --desc_path FB15k_mid2description.txt --name_path FB15k_mid2name.txt --output_dir data/FB15k237_processed

python train_dgsaf.py --data_dir data/FB15k237_processed --output_dir outputs/dgsaf --batch_size 32 --epochs 20 --text_dim 200 --complex_dim 400
