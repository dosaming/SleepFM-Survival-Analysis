import os, pickle
import numpy as np


SURV_DIR = "/ssd/kdpark/sleepfm-codebase/outputs_shhs_list/emb_pickles"   # survival_{split}.pickle 위치
INFO_PKL = "/ssd/kdpark/sleepfm-codebase/sleepfm/survival_all.pickle"     # info 전체 피클(x, subject_ids)
FEAT_TXT = "/ssd/kdpark/sleepfm-codebase/sleepfm/feature_names.txt"       # info x의 (원-핫 이후) 컬럼명
OUT_DIR  = "/ssd/kdpark/sleepfm-codebase/sleepfm/merged_final"

SPLITS = ["train", "valid", "test"]
INFO_PREFIXES = ["nsrr_age", "nsrr_sex", "nsrr_race", "nsrr_ethnicity", "nsrr_bmi"]

os.makedirs(OUT_DIR, exist_ok=True)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def select_info_columns(info_feat_names, prefixes):
    idx = [i for i, name in enumerate(info_feat_names)
           if any(name == p or name.startswith(p + "_") for p in prefixes)]
    if not idx:
        raise ValueError("해당하는 열이 없습니다.")
    return idx

def merge_one(split):
    surv_path = os.path.join(SURV_DIR, f"survival_{split}.pickle")
    out_path  = os.path.join(OUT_DIR,  f"survival_{split}_merged.pickle")

    # survival_split.pickle에서 ids, t, e, X_embed 모두 가져옴
    d_surv = load_pickle(surv_path)
    ids = np.asarray(d_surv["subject_ids"]).astype(str)
    X_embed = np.asarray(d_surv["x"], dtype=np.float32)
    t = np.asarray(d_surv["t"], dtype=np.float32)
    e = np.asarray(d_surv["e"], dtype=np.int32)

    assert X_embed.ndim == 2 and len(ids) == X_embed.shape[0] == len(t) == len(e), \
        f"{surv_path}: 차원이 맞지 않습니다."

    #
    with open(FEAT_TXT, "r") as f:
        info_feat_names = [ln.strip() for ln in f if ln.strip()]
    keep_idx = select_info_columns(info_feat_names, INFO_PREFIXES)

    d_info = load_pickle(INFO_PKL)
    ids_info = np.asarray(d_info["subject_ids"]).astype(str)
    X_info_full = np.asarray(d_info["x"], dtype=np.float32)
    assert X_info_full.ndim == 2 and X_info_full.shape[1] == len(info_feat_names)
    X_info = X_info_full[:, keep_idx].astype(np.float32)

    # ids 기준 inner-join 정렬 
    idx_info = {sid: i for i, sid in enumerate(ids_info)}
    rows_surv, rows_info = [], []
    missing = 0
    for i, sid in enumerate(ids):
        j = idx_info.get(sid)
        if j is not None:
            rows_surv.append(i); rows_info.append(j)
        else:
            missing += 1
    rows_surv = np.asarray(rows_surv, dtype=int)
    rows_info = np.asarray(rows_info, dtype=int)
    assert len(rows_surv) > 0, f"{split}: info와 매칭된 ID가 없습니다."

    ids_m = ids[rows_surv]
    X_em_m = X_embed[rows_surv]
    X_in_m = X_info[rows_info]
    t_m = t[rows_surv]
    e_m = e[rows_surv]

  
    X_merged = np.concatenate([X_em_m, X_in_m], axis=1).astype(np.float32)
    mask = np.isfinite(X_merged).all(axis=1) & np.isfinite(t_m) & np.isfinite(e_m)
    X_merged = X_merged[mask]
    t_m = t_m[mask].astype(np.float32)
    e_m = e_m[mask].astype(np.int32)
    ids_m = ids_m[mask]

    out = {
        "x": X_merged,
        "t": t_m,
        "e": e_m,
        "subject_ids": ids_m,
        "feature_groups": {
            "embed_dim": int(X_em_m.shape[1]),
            "info_selected": [info_feat_names[i] for i in keep_idx],
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] {split}: saved -> {out_path}")
    print(f"     N={X_merged.shape[0]}, D_embed={X_em_m.shape[1]}, D_info={X_in_m.shape[1]}, D_total={X_merged.shape[1]}")
    print(f"     events dist: {dict(zip(*np.unique(e_m, return_counts=True)))}")
    if missing:
        print(f"     (info에서 못 찾은 ID: {missing}명)")

def main():
    for split in SPLITS:
        merge_one(split)

if __name__ == "__main__":
    main()
