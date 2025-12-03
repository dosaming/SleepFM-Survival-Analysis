# Environment  
dataset.py의 from config_shhs_125 / new 로 설정 확인 필요  
config_shhs_new : X_EOG_diff 전처리한 데이터 경로로 저장한 config.py  

# Extract pretraining dataset  
'''python 0_extract_pretraining_data_shhs.py \
--shhs_edf_dir "/#_SHHS/polysomnography/edfs/shhs1" \
--shhs_xml_dir "/#_SHHS/polysomnography/annotations-events-profusion/shhs1" \
--target_sampling_rate 125 \
--num_threads 8'''  


# Prepare Dataset (125Hz)  
'''python 1_prepare_dataset_shhs.py \
--dataset_dir shhs_segments_125 \
--train_frac 0.70 \
--valid_frac 0.10 \
--test_size 0.20 \
--num_threads 8 \
--random_state 42  
Split sizes → Train: 3047, Valid: 1306, Test: 1088  

# Generate Embedding  
python 3_generate_embed_pretraining_shhs.py \
outputs_shhs_125 \
--dataset_dir shhs_segments_125 \
--dataset_file dataset_events_-1.pickle \
--batch_size 64 \
--num_workers 0 \
--splits train,valid,test

# Classification Eval Pretraining  
python 4_classification_eval_pretraining_shhs.py \
--output_file outputs_shhs_125 \
--dataset_dir shhs_segments_125 \
--modality_type sleep_stages \
--model_name logistic \
--max_iter 1000  

# Results  

<img width="364" height="156" alt="image" src="https://github.com/user-attachments/assets/458f786e-ba1d-4f97-a5e0-34afc72b67c8" />  

# Fine Encoder  

python fine_encoder_final_shhs.py  
python fine_encoder_wce_shhs.py  

# Fine Encoder Results  
<img width="364" height="156" alt="image" src="https://github.com/user-attachments/assets/c9a3e4bf-9f46-4c5a-9288-c840f65715c2" />





