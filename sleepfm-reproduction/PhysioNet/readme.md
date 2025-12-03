# Settings
```
cd /ssd/kdpark/sleepfm-codebase  
conda activate sleepfm_env
```

# 0_extract_pretraining_data  
```
python 0_extract_pretraining_data.py
```

# 1_prepare_dataset  
physionet split: train 742, valid 50, test 199  
```
python 1_prepare_dataset
```
* 버전 오류 시 sleep_env에서  
   ```
   export LD_LIBRARY_PATH= /sleepfm_env/lib:$LD_LIBRARY_PATH
   ```


# 3_generate_embed_pretraining  
(dataset.py에서 참조하는 config 이름 확인 필요)  
```
python 3_generate_embed_pretraining.py \
my_run_final \
--dataset_dir physionet_final \
--dataset_file dataset_events_-1.pickle \
--splits train,valid,test
```


# 4_classification_eval_pretraining  
```
python 4_classification_eval_pretraining.py \
--output_file my_run_final \
--dataset_dir physionet_final \
--modality_type sleep_stages \
--model_name logistic \
--max_iter 1000  
```

# Results  

<img width="364" height="156" alt="image" src="https://github.com/user-attachments/assets/6677fa87-f31a-4493-8720-eea725fd22a8" />  

# Fine Encoder  
encoder: unfreeze + CrossEntropy로 end-to-end 학습  
```
python fine_encoder_final.py
```

# Fine Encoder WCE  
```
ulimit -n 65536  
python fine_encoder_wce0.py
```

# WCE Results  
<img width="364" height="156" alt="image" src="https://github.com/user-attachments/assets/00df3315-3f2c-4eae-8aba-a3e0fa178888" />






