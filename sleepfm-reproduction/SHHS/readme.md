# Environment  
/home/kdpark/sleepfm-codebase/sleepfm/model/dataset.py의 from config_shhs_125로 설정 확인  

# Extract pretraining dataset  
python /home/kdpark/sleepfm-codebase/sleepfm/0_extract_pretraining_data_shhs.py \
--shhs_edf_dir "/ssd/datasets/sleep/#_SHHS/polysomnography/edfs/shhs1" \
--shhs_xml_dir "/ssd/datasets/sleep/#_SHHS/polysomnography/annotations-events-profusion/shhs1" \
--target_sampling_rate 125 \
--num_threads 8  


# Prepare Dataset (125Hz)  
python /home/kdpark/sleepfm-codebase/sleepfm/1_prepare_dataset_shhs.py \
--dataset_dir /ssd/kdpark/sleepfm-codebase/shhs_segments_125 \
--train_frac 0.70 \
--valid_frac 0.10 \
--test_size 0.20 \
--num_threads 8 \
--random_state 42  
Split sizes → Train: 3047, Valid: 1306, Test: 1088  

# Generate Embedding  
python 3_generate_embed_pretraining_shhs.py \
/ssd/kdpark/sleepfm-codebase/outputs_shhs_125 \
--dataset_dir /ssd/kdpark/sleepfm-codebase/shhs_segments_125 \
--dataset_file dataset_events_-1.pickle \
--batch_size 64 \
--num_workers 0 \
--splits train,valid,test

# Classification Eval Pretraining  
python /home/kdpark/sleepfm-codebase/sleepfm/4_classification_eval_pretraining_shhs.py \
--output_file /ssd/kdpark/sleepfm-codebase/outputs_shhs_125 \
--dataset_dir /ssd/kdpark/sleepfm-codebase/shhs_segments_125 \
--modality_type sleep_stages \
--model_name logistic \
--max_iter 1000  

# Results  

<img width="364" height="156" alt="image" src="https://github.com/user-attachments/assets/458f786e-ba1d-4f97-a5e0-34afc72b67c8" />




