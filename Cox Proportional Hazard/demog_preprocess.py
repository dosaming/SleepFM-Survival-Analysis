import numpy as np
import pandas as pd
import pickle
import os

csv_path = "/ssd/kdpark/sleepfm-codebase/sleepfm/cox_df.csv"
out_dir  = "/ssd/kdpark/sleepfm-codebase/sleepfm"
os.makedirs(out_dir, exist_ok=True)

cox_df = pd.read_csv(csv_path)

feat_cols = ["nsrr_age", "nsrr_sex", "nsrr_race", "nsrr_ethnicity", "nsrr_bmi"]

df_feat = cox_df[feat_cols].copy()
cat_cols = [c for c in feat_cols if df_feat[c].dtype == "object"]
num_cols = [c for c in feat_cols if c not in cat_cols]

X_num = df_feat[num_cols].apply(pd.to_numeric, errors="coerce") if num_cols else pd.DataFrame(index=df_feat.index)
X_cat = pd.get_dummies(df_feat[cat_cols].astype(str), prefix=cat_cols, dummy_na=False) if cat_cols else pd.DataFrame(index=df_feat.index)
X_df = pd.concat([X_num, X_cat], axis=1)

t = cox_df["duration"].astype(float).values
e = cox_df["event"].astype(int).values

X = X_df.values.astype(np.float32)
mask = np.isfinite(X).all(axis=1) & np.isfinite(t) & np.isfinite(e)
X, t, e = X[mask], t[mask].astype(np.float32), e[mask].astype(np.int32)

ids = cox_df.loc[mask, "nsrrid"].astype(str).values if "nsrrid" in cox_df.columns else None

payload = {"x": X, "t": t, "e": e}
if ids is not None:
    payload["subject_ids"] = ids

pickle_path = os.path.join(out_dir, "survival_all.pickle")
with open(pickle_path, "wb") as f:
    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

feat_name_path = os.path.join(out_dir, "feature_names.txt")
with open(feat_name_path, "w") as f:
    for c in X_df.columns:
        f.write(f"{c}\n")

print(f"[OK] Saved pickle: {pickle_path}")
print(f"[OK] Saved feature names: {feat_name_path}")
print(f"Shape: X={X.shape}, t={t.shape}, e={e.shape}")
