# Environment  
scipy 설치 취소 후 버전 맞춰서 다시 설치  
```
pip install scipy==1.9.3
```

# Data Preprocess  
```
python cox_preprocess_1.py \
  --dataset_file shhs_segments_125/dataset_events_-1.pickle \
  --split valid \
  --modality sleep_stages \
  --ckpt_path /outputs_shhs_list/runs/250829_124007_sleep_stages/ft_last.pt \
  --out_dir /outputs_shhs_list/emb_pickles
```
  
```
python cox_preprocess_2.py \
  --embeddings_pickle outputs_shhs_list/emb_pickles/embeddings_train.pickle \
  --dataset_file     shhs_segments_125/dataset_events_-1.pickle \
  --split            train \
  --modality         sleep_stages \
  --events_csv       shhs-cvd-events-dataset-0.20.0.csv \
  --out_dir          outputs_shhs_list/emb_pickles  
```
```    
python merged_final.py  
```


# CoxMLP  
```
python CoxMLP.py \
  --train_pickle /merged_final/survival_train_merged.pickle \
  --valid_pickle /merged_final/survival_valid_merged.pickle \
  --test_pickle  /merged_final/survival_test_merged.pickle
--epochs 400 --lr 5e-4 --l2 5e-3 \
--hidden 64 --dropout 0.4 --val_freq 5 --ties breslow --grad_clip 1.0  
```

# Code Explanation
데이터 준비  
사건 발생 시간 역순으로 데이터 정렬 → 분모 빠르게 얻기 위해 누적합 cumsum 사용  

기본적인 렐루 함수로 구성된 MLP 구축  

risk = self.risk(deterministic) : 각 환자의 로그 위험비. 신경망 최종 출력  

uncensored_likelihood = risk.T - log_risk : 콕스의 likelihood 코드로 수현  

neg_likelihood = -T.sum(censored_likelihood) / num_observed_events  
:   
uncensored_likelihood에 이벤트 발생 여부를 곱해서 실제로 이벤트 발생한 환자만 선택되도록  

# Data Information  
최종적으로 survival 데이터와 info 데이터(나이, 성별, BMI 등)가 모두 존재하는 환자 수  
train: 3444명  
valid: 88명  
test: 1534명  
-> info data 없는 31명 제외  

D_embed (psg) = 512  
D_info = 5 ["nsrr_age", "nsrr_sex", "nsrr_race", "nsrr_ethnicity", "nsrr_bmi"]  
event 발생 771건, event 미발생 2673건  


# survival_all.pickle  
<img width="1074" height="532" alt="image" src="https://github.com/user-attachments/assets/45ea9278-bc1c-4cbc-9ff9-917e8cd1dbb8" />  


# C-Index Result  
train 0.8244  
valid 0.7391  
test 0.7114  
