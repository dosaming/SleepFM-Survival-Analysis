

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/ssd/kdpark/sleepfm-codebase')  # sleepfm 패키지 루트

import os
import re
import glob
import json
import time
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[SAVE] {path}")

def parse_datetime_utc(s):
    return pd.to_datetime(s, errors="coerce", utc=True)

def preprocess_events(events_csv, time_origin, id_col="nsrrid", event_dt_col="event_dt"):
    """
    CSV에서 동일 nsrrid의 최초 event만 유지해 event=1로 마킹.
    event_dt가 '일수(숫자)'면 time_origin + days 로 절대시각 생성,
    날짜/날짜문자열이면 그대로 파싱.
    반환: DataFrame[nsrrid(str), duration_dt(UTC), event=1]
    """
    df = pd.read_csv(events_csv)
    df.columns = [c.lower() for c in df.columns]

    cand_id = [id_col, "subject_id", "id"]
    cand_dt = [event_dt_col, "event_date", "event_time", "event_datetime"]

    def pick(options):
        for o in options:
            if o in df.columns:
                return o
        return None

    id_c = pick(cand_id) or id_col
    dt_c = pick(cand_dt) or event_dt_col
    if id_c not in df.columns:
        raise ValueError(f"ID column not found (tried {cand_id}), have {df.columns.tolist()}")
    if dt_c not in df.columns:
        raise ValueError(f"Date/time column not found (tried {cand_dt}), have {df.columns.tolist()}")

    # 1) event_dt가 숫자(= time_origin으로부터 경과일)인지 판별
    num = pd.to_numeric(df[dt_c], errors="coerce")
    if num.notna().sum() >= 0.9 * len(df):
        # 대부분 숫자면 '일수'로 간주
        if time_origin == "auto":
            raise ValueError(
                "events_csv의 event_dt가 '경과일(숫자)' 형식입니다. "
                "--time_origin=YYYY-MM-DD 를 명시해 주세요."
            )
        origin = parse_datetime_utc(time_origin)
        df["duration_dt"] = origin + pd.to_timedelta(num.astype(float), unit="D")
    else:
        # 날짜/날짜문자열로 간주
        df["duration_dt"] = parse_datetime_utc(df[dt_c])

    df = df.dropna(subset=["duration_dt"])

    earliest = (
        df.sort_values([id_c, "duration_dt"])
          .groupby(id_c, as_index=False)
          .agg({"duration_dt": "first"})
    )
    out = earliest.rename(columns={id_c: "nsrrid"})
    out["event"] = 1
    out["nsrrid"] = out["nsrrid"].astype(str).str.strip()
    return out

def build_censor_df(all_subject_ids, censor_csv=None, censor_end_date=None):
    sids = pd.DataFrame({"nsrrid": list(map(str, all_subject_ids))})
    sids["nsrrid"] = sids["nsrrid"].astype(str).str.strip()

    if censor_csv:
        cdf = pd.read_csv(censor_csv)
        cdf.columns = [c.lower() for c in cdf.columns]
        if "nsrrid" not in cdf.columns:
            raise ValueError("censor_csv must contain 'nsrrid'")
        for k in ["censor_dt", "censor_date", "end_dt", "end_date"]:
            if k in cdf.columns:
                dt_col = k
                break
        else:
            raise ValueError("censor_csv must contain a censor date column (e.g., 'censor_dt')")
        cdf[dt_col] = parse_datetime_utc(cdf[dt_col])
        cdf = cdf.rename(columns={dt_col: "censor_dt"})
        cdf = cdf[["nsrrid", "censor_dt"]]
        out_df = sids.merge(cdf, on="nsrrid", how="left")
    else:
        if censor_end_date is None:
            raise ValueError("Provide either --censor_csv or --censor_end_date")
        dt = parse_datetime_utc(censor_end_date)
        out_df = sids.assign(censor_dt=dt)

    out_df["nsrrid"] = out_df["nsrrid"].astype(str).str.strip()
    return out_df

def days_since_origin(dt_series, origin=None):
    if origin is None:
        origin = pd.to_datetime(dt_series.min(), utc=True)
    delta = (dt_series - origin).dt.total_seconds() / 86400.0
    return delta.astype(np.float32), origin

def extract_nsrrid_from_path(p: str) -> str:
    p = str(p)
    parent = os.path.basename(os.path.dirname(p))
    fname  = os.path.basename(p)
    m = re.search(r"shhs\d+-(\d{5,7})", parent)
    if m: return m.group(1)
    m = re.search(r"(\d{5,7})", fname)
    if m: return m.group(1)
    m = re.search(r"(\d{5,7})", p)
    if m: return m.group(1)
    return fname

def try_paths_from_dataset(dataset_file, split, modality):
    try:
        from sleepfm.model.dataset import EventDatasetSupervised
        ds = EventDatasetSupervised(root=dataset_file, split=split, modality_type=modality)
        list_attrs = ["file_paths","files","paths","segment_paths","segments","npy_paths","x_paths","records","edf_paths"]
        for attr in list_attrs:
            if hasattr(ds, attr):
                val = getattr(ds, attr)
                if val is not None and hasattr(val, "__len__") and len(val) == len(ds):
                    return list(map(str, val))
        df_attrs = ["segments_df","df","meta_df"]
        col_candidates = ["path","filepath","file_path","npy","npy_path","file","fullpath"]
        for attr in df_attrs:
            if hasattr(ds, attr):
                df = getattr(ds, attr)
                if df is not None and hasattr(df, "columns"):
                    cols = [c.lower() for c in df.columns]
                    for c in col_candidates:
                        if c in cols:
                            series = df[c].astype(str).tolist()
                            if len(series) == len(ds):
                                return series
    except Exception:
        pass
    return None

def paths_from_glob_or_guess(dataset_file, modality, segments_glob=None):
    if segments_glob:
        paths = sorted(glob.glob(segments_glob))
        if not paths:
            raise RuntimeError(f"--segments_glob 수집 결과가 비었습니다: {segments_glob}")
        return paths
    root = os.path.dirname(dataset_file)
    sub_map = {
        "sleep_stages": "X_Sleep_Stages",
        "respiratory" : "X_Respiratory",
        "ekg"         : "X_EKG",
    }
    sub = sub_map.get(modality, "")
    guess_root = os.path.join(root, sub) if sub else root
    guess_glob = os.path.join(guess_root, "*", "*.npy")
    paths = sorted(glob.glob(guess_glob))
    if not paths:
        raise RuntimeError(
            "경로 자동 추정 실패. --segments_glob 로 명시해 주세요.\n"
            f"(자동 시도 패턴: {guess_glob})"
        )
    return paths

def get_paths_in_embedding_order(dataset_file, split, modality, segments_glob=None):
    paths = try_paths_from_dataset(dataset_file, split, modality)
    if paths is not None:
        return paths
    return paths_from_glob_or_guess(dataset_file, modality, segments_glob)

def aggregate_subject_embeddings(Z, ids, how="mean"):
    groups = defaultdict(list)
    for z, sid in zip(Z, ids):
        groups[str(sid)].append(z)
    subject_ids, X_subject = [], []
    for sid, vecs in groups.items():
        mat = np.stack(vecs, axis=0)
        if how == "mean":
            pooled = mat.mean(axis=0)
        elif how == "median":
            pooled = np.median(mat, axis=0)
        elif how == "max":
            pooled = mat.max(axis=0)
        else:
            raise ValueError(f"Unknown pooling: {how}")
        subject_ids.append(sid)
        X_subject.append(pooled)
    return np.stack(X_subject, axis=0).astype(np.float32), np.array(subject_ids, dtype=object)

def main():
    parser = argparse.ArgumentParser(
        description="(Post-Embedding) Load embeddings → infer nsrrid from paths → subject pooling → build survival x,t,e"
    )
    parser.add_argument("--embeddings_pickle", type=str, required=True)
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train","valid","test"])
    parser.add_argument("--modality", type=str, default="sleep_stages",
                        choices=["sleep_stages","respiratory","ekg","combined"])
    parser.add_argument("--events_csv", type=str, required=True)
    parser.add_argument("--censor_csv", type=str, default=None)
    # 기본 관찰창
    parser.add_argument("--censor_end_date", type=str, default="2011-12-31",
                        help="Default censor date for non-event subjects (YYYY-MM-DD).")
    parser.add_argument("--time_origin", type=str, default="1995-11-01",
                        help="Origin date for survival time (YYYY-MM-DD) or 'auto'.")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean","median","max"])
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--segments_glob", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 임베딩 로드
    with open(args.embeddings_pickle, "rb") as f:
        data = pickle.load(f)
    Z = data["X"]
    ids = data.get("ids", None)
    print(f"[INFO] loaded embeddings: X={Z.shape}, ids={'present' if ids is not None else 'missing'}")

    # 2) ids 없으면 경로에서 추론
    if ids is None or any(v is None for v in (ids.tolist() if isinstance(ids, np.ndarray) else ids)):
        print("[INFO] inferring nsrrid from paths …")
        paths = get_paths_in_embedding_order(
            args.dataset_file, args.split, args.modality, segments_glob=args.segments_glob
        )
        if len(paths) != len(Z):
            raise ValueError(
                f"len(paths)={len(paths)} != len(embeddings)={len(Z)}\n"
                "→ 임베딩 생성과 동일 split/modality/글롭인지 확인하세요."
            )
        inferred = [extract_nsrrid_from_path(p) for p in paths]
        ids = np.array(inferred, dtype=object)
        data["ids"] = ids
        patched = os.path.join(args.out_dir, f"embeddings_{args.split}_with_ids.pickle")
        save_pickle(data, patched)
        print(f"[INFO] ids injected into embeddings → {patched}")

    # 3) subject 단위 풀링
    X_subject, subject_ids = aggregate_subject_embeddings(Z, ids, how=args.pooling)
    print(f"[INFO] subject-level: X={X_subject.shape}, n_subjects={len(subject_ids)}")

    # 4) 이벤트/검열 테이블
    events_df = preprocess_events(args.events_csv, time_origin=args.time_origin)  # ★ 변경: time_origin 전달
    censor_df = build_censor_df(subject_ids, args.censor_csv, args.censor_end_date)

    events_df["nsrrid"] = events_df["nsrrid"].astype(str).str.strip()
    censor_df["nsrrid"] = censor_df["nsrrid"].astype(str).str.strip()
    all_df = pd.DataFrame({"nsrrid": subject_ids.astype(str)})
    all_df["nsrrid"] = all_df["nsrrid"].astype(str).str.strip()

    # 5) 관찰창 필터: [time_origin, censor_end_date]
    if args.time_origin != "auto":
        lo = parse_datetime_utc(args.time_origin)
    else:
        lo = pd.to_datetime("1900-01-01", utc=True)
    hi = parse_datetime_utc(args.censor_end_date) if args.censor_end_date else pd.to_datetime("2100-01-01", utc=True)
    before = len(events_df)
    events_df = events_df[(events_df["duration_dt"] >= lo) & (events_df["duration_dt"] <= hi)].copy()
    after = len(events_df)
    if before != after:
        print(f"[INFO] window filter applied: kept {after}/{before} events within [{lo.date()} ~ {hi.date()}]")

    # 6) merge & duration_dt 선택
    all_df = all_df.merge(events_df[["nsrrid","duration_dt","event"]], on="nsrrid", how="left")
    all_df["event"] = all_df["event"].fillna(0).astype(int)
    all_df = all_df.merge(censor_df, on="nsrrid", how="left")
    all_df["duration_dt"] = np.where(all_df["event"].eq(1), all_df["duration_dt"], all_df["censor_dt"])

    if args.censor_csv is None and args.censor_end_date is not None:
        all_df.loc[all_df["event"].eq(0), "duration_dt"] = parse_datetime_utc(args.censor_end_date)

    if all_df["duration_dt"].isna().any():
        missing = all_df[all_df["duration_dt"].isna()]["nsrrid"].tolist()
        raise ValueError(
            f"duration_dt could not be determined for some subjects. "
            f"검열 정보( --censor_end_date 또는 --censor_csv )를 확인하세요. 예: {missing[:8]} … (총 {len(missing)})"
        )

    # 7) 시간 수치화
    if args.time_origin == "auto":
        t_days, origin = days_since_origin(all_df["duration_dt"])
    else:
        origin = parse_datetime_utc(args.time_origin)
        t_days, _ = days_since_origin(all_df["duration_dt"], origin=origin)

    e = all_df["event"].values.astype(np.int32)
    surv_dict = {
        "x": X_subject.astype(np.float32),
        "t": t_days,
        "e": e,
        "subject_ids": all_df["nsrrid"].tolist(),
        "time_origin": origin.isoformat(),
        "duration_unit": "days_since_origin",
    }
    surv_pkl = os.path.join(args.out_dir, f"survival_{args.split}.pickle")
    save_pickle(surv_dict, surv_pkl)

    meta_json = os.path.join(args.out_dir, f"survival_{args.split}.json")
    with open(meta_json, "w") as fp:
        json.dump({
            "x_shape": list(surv_dict["x"].shape),
            "n_subjects": len(surv_dict["subject_ids"]),
            "columns": {"subject_id": "nsrrid", "duration_days": "t", "event": "e"},
            "pooling": args.pooling,
            "time_origin": origin.isoformat(),
            "src": {
                "embeddings_pickle": args.embeddings_pickle,
                "events_csv": args.events_csv,
                "censor_csv": args.censor_csv,
                "censor_end_date": args.censor_end_date,
            }
        }, fp, ensure_ascii=False, indent=2)
    print(f"[SAVE] {meta_json}")
    print("\nDone ")
    print(f"- embeddings used : {args.embeddings_pickle}")
    print(f"- survival output : {surv_pkl}  (x,t,e,subject_ids)")

if __name__ == "__main__":
    main()
