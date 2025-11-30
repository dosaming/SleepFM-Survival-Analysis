# Settins
cd /ssd/kdpark/sleepfm-codebase  
conda activate sleepfm_env  

# 0_extract_pretraining_data  
python 0_extract_pretraining_data.py

# 1_prepare_dataset  
physionet split: train 742, valid 50, test 199  
python 1_prepare_dataset  
* 버전 오류 시 sleep_env에서 
   export LD_LIBRARY_PATH=/ssd/kdpark/anaconda3/envs/sleepfm_env/lib:$LD_LIBRARY_PATH


# 3_generate_embed_pretraining  

python /home/kdpark/sleepfm-codebase/sleepfm/3_generate_embed_pretraining.py \
  my_run \
  --data_dir /ssd/kdpark/sleepfm-codebase/physionet_segments \
  --dataset_file dataset_events_-1.pickle \
  --splits train,valid,test

# 4_classification_eval_pretraining  
python /ssd/kdpark/sleepfm-codebase/sleepfm/4_classification_eval_pretraining.py \
--output_file my_run_final \
--dataset_dir /ssd/kdpark/sleepfm-codebase/physionet_final \
--modality_type sleep_stages \
--model_name logistic \
--max_iter 1000
