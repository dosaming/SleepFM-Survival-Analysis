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
