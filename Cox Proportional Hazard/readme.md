# enviroment  
pip install scipy==1.9.3

# CoxMLP  
python /ssd/kdpark/sleepfm-codebase/sleepfm/CoxMLP.py \
  --train_pickle /ssd/kdpark/sleepfm-codebase/sleepfm/merged_final/survival_train_merged.pickle \
  --valid_pickle /ssd/kdpark/sleepfm-codebase/sleepfm/merged_final/survival_valid_merged.pickle \
  --test_pickle  /ssd/kdpark/sleepfm-codebase/sleepfm/merged_final/survival_test_merged.pickle
