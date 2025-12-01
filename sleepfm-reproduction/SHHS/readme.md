# Environment  

# Extract pretraining dataset  
python /home/kdpark/sleepfm-codebase/sleepfm/0_extract_pretraining_data_shhs.py \
--shhs_edf_dir "/ssd/datasets/sleep/#_SHHS/polysomnography/edfs/shhs1" \
--shhs_xml_dir "/ssd/datasets/sleep/#_SHHS/polysomnography/annotations-events-profusion/shhs1" \
--target_sampling_rate 125 \
--num_threads 8  


# Prepare Dataset (125Hz)  
python /home/kdpark/sleepfm-codebase/sleepfm/1_prepare_dataset_shhs125.py \
--dataset_dir /ssd/kdpark/sleepfm-codebase/shhs_segments_125 \
--train_frac 0.70 \
--valid_frac 0.10 \
--test_size 0.20 \
--num_threads 8 \
--random_state 42  
Split sizes → Train: 3047, Valid: 1306, Test: 1088  

# 


