

import os
import pickle
import argparse
from loguru import logger
from tqdm import tqdm
import numpy as np
import random
import multiprocessing

from config_shhs_125 import (
    LABEL_MAP, PATH_TO_PROCESSED_DATA,
    EVENT_TO_ID, INV_LABELS_DICT
)

def normalize_label(label_raw):

    if isinstance(label_raw, int):
        name = INV_LABELS_DICT.get(label_raw)
        if name is None:
            return None, None
        return label_raw, name

    if isinstance(label_raw, str) and label_raw.isdigit():
        lid = int(label_raw)
        name = INV_LABELS_DICT.get(lid)
        if name is None:
            return None, None
        return lid, name

    if isinstance(label_raw, str):
        
        if label_raw in LABEL_MAP:
            std = LABEL_MAP[label_raw]          
            lid = EVENT_TO_ID.get(std)
            if lid is None:
                return None, None
            return lid, std

        if '|' in label_raw:
            base = label_raw.split('|', 1)[0].replace(' sleep', '').strip()
            std = LABEL_MAP.get(base, base)
            lid = EVENT_TO_ID.get(std)
            if lid is None:
                return None, None
            return lid, std

        low = label_raw.lower()
        if low in LABEL_MAP:
            std = LABEL_MAP[low]
            lid = EVENT_TO_ID.get(std)
            if lid is None:
                return None, None
            return lid, std

    return None, None

def parallel_prepare_data(args_tuple):
    mrns, dataset_dir, mrn_train, mrn_valid, mrn_test = args_tuple

    data_dict = {"train": [], "valid": [], "test": []}
    empty_label_dict_counts = 0
    path_to_Y = os.path.join(dataset_dir, "Y")
    path_to_X = os.path.join(dataset_dir, "X")

    for mrn in tqdm(mrns, leave=False, desc="patients(shard)"):
        one_patient = {mrn: {}}
        path_to_patient = os.path.join(path_to_X, mrn)
        path_to_label = os.path.join(path_to_Y, f"{mrn}.pickle")

        if mrn in mrn_train:
            split_name = "train"
        elif mrn in mrn_valid:
            split_name = "valid"
        elif mrn in mrn_test:
            split_name = "test"
        else:
            logger.warning(f"{mrn} not in any split (skip).")
            continue

        if not os.path.exists(path_to_label):
            logger.info(f"{mrn} label pickle not found (skip).")
            continue
        if not os.path.exists(path_to_patient):
            logger.info(f"{mrn} X folder not found (skip).")
            continue

        with open(path_to_label, "rb") as f:
            labels_dict = pickle.load(f)
        if not labels_dict:
            empty_label_dict_counts += 1
            logger.info(f"{mrn} label_dict is empty (skip).")
            continue

        for fname in sorted(os.listdir(path_to_patient)):
            if not fname.endswith(".npy"):
                continue
            fpath = os.path.join(path_to_patient, fname)

            if fname not in labels_dict:
                logger.warning(f"{mrn}: missing label for {fname} (skip).")
                continue

            label_raw = labels_dict[fname]  # int or str
            label_id, label_name = normalize_label(label_raw)
            if label_id is None:
                logger.warning(f"{mrn}: cannot normalize label for {fname} -> {label_raw} (skip).")
                continue

            if label_name not in one_patient[mrn]:
                one_patient[mrn][label_name] = []
            one_patient[mrn][label_name].append(fpath)

        if any(len(v) > 0 for v in one_patient[mrn].values()):
            data_dict[split_name].append(one_patient)

    logger.info(f"Empty label dicts in this worker: {empty_label_dict_counts}")
    return data_dict

def split_patients(mrns, train_frac, valid_frac, test_spec, seed):

    rng = np.random.default_rng(seed)
    mrns = np.array(sorted(mrns))  
    rng.shuffle(mrns)
    n = len(mrns)

  
    if test_spec <= 1.0:
        n_test = max(1, int(round(n * float(test_spec))))
    else:
        n_test = int(test_spec)
        n_test = max(1, min(n_test, n - 2))  

    test = set(mrns[:n_test])
    remain = mrns[n_test:]
    m = len(remain)

    n_train = max(1, int(round(m * float(train_frac))))
    n_valid = m - n_train
    target_valid = int(round(m * float(valid_frac)))
    if n_valid == 0 and target_valid > 0 and n_train > 1:
        n_train -= 1
        n_valid += 1

    train = set(remain[:n_train])
    valid = set(remain[n_train:])

    return train, valid, test

def main():
    parser = argparse.ArgumentParser(description="Build SHHS dataset splits & event lists (30s epochs @ 125 Hz)")
    parser.add_argument("--dataset_dir", type=str, default=None, help="PATH_TO_PROCESSED_DATA override")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--min_sample", type=int, default=-1,
                        help="Per (patient,label) max samples; -1 uses all")
    # 5804명: train 70%, valid 10%, test 20%
    parser.add_argument("--train_frac", type=float, default=0.70,
                        help="Fraction of patients for train (default 0.70)")
    parser.add_argument("--valid_frac", type=float, default=0.10,
                        help="Fraction of remaining patients for valid (default 0.10)")
    parser.add_argument("--test_size", type=float, default=0.20,
                        help="If <=1: fraction, if >1: absolute count (default 0.20)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or PATH_TO_PROCESSED_DATA
    path_to_X = os.path.join(dataset_dir, "X")
    mrns = sorted([d for d in os.listdir(path_to_X) if os.path.isdir(os.path.join(path_to_X, d))])

    if args.debug:
        logger.info("Running in DEBUG mode (first 100 MRNs)")
        mrns = mrns[:100]

    n_total = len(mrns)
    logger.info(f"Number of patients (MRNs): {n_total}")
    if n_total < 100:
        logger.warning("MRN 수가 너무 적습니다. 분할 비율/절대 수를 조정하세요.")

    
    mrn_train, mrn_valid, mrn_test = split_patients(
        mrns,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_spec=args.test_size,     
        seed=args.random_state,
    )
    logger.info(f"Split sizes → Train: {len(mrn_train)}, Valid: {len(mrn_valid)}, Test: {len(mrn_test)}")

    num_threads = max(1, args.num_threads)
    shards = np.array_split(np.array(mrns), num_threads)
    tasks = [(list(s), dataset_dir, mrn_train, mrn_valid, mrn_test) for s in shards]

    with multiprocessing.Pool(num_threads) as pool:
        partial = list(pool.imap_unordered(parallel_prepare_data, tasks))

    dataset = {"train": [], "valid": [], "test": []}
    for d in partial:
        for k, v in d.items():
            dataset[k].extend(v)

    for k in dataset:
        dataset[k] = sorted(dataset[k], key=lambda x: list(x.keys())[0])

    out1 = os.path.join(dataset_dir, "dataset.pickle")
    with open(out1, "wb") as f:
        pickle.dump(dataset, f)
    logger.info(f"Saved dataset dict to {out1}")

    rng = random.Random(args.random_state)
    dataset_event = {}
    for split, split_data in tqdm(dataset.items(), total=len(dataset), desc="build event tuples"):
        sampled = []
        for item in split_data:
            mrn = list(item.keys())[0]
            pdata = item[mrn]  
            for label_name, paths in pdata.items():
                if args.min_sample == -1:
                    chosen = paths
                else:
                    if len(paths) > args.min_sample:
                        chosen = rng.sample(paths, args.min_sample)
                    else:
                        chosen = paths
                sampled.extend([(p, label_name) for p in chosen])
        rng.shuffle(sampled)
        dataset_event[split] = sampled

    out2 = os.path.join(dataset_dir, "dataset_events_-1.pickle")
    with open(out2, "wb") as f:
        pickle.dump(dataset_event, f)
    logger.info(f"Saved event list to {out2}")

if __name__ == "__main__":
    main()
