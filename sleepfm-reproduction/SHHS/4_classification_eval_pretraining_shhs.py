
import os
import sys
import pickle
import argparse
import collections

import numpy as np
from tqdm import tqdm
from loguru import logger

from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix
)
from xgboost import XGBClassifier


sys.path.append('/ssd/kdpark/sleepfm-codebase')

# from utils import train_model
from sleepfm.model.dataset import EventDataset as Dataset  

import config_shhs
from config_shhs import (
    MODALITY_TYPES, CLASS_LABELS, LABELS_DICT, PATH_TO_PROCESSED_DATA,
    LABEL_MAP, EVENT_TO_ID
)


def resolve_output_path(dataset_dir: str, output_file: str) -> str:
    if os.path.isabs(output_file):
        return output_file
    return os.path.join(dataset_dir, output_file)

def concat_modal_embeddings(arr):
    """3개 모달리티 임베딩을 가로로 연결하기"""
    if isinstance(arr, list):
        return np.concatenate(arr, axis=1)
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 3:
        return np.concatenate([arr[0], arr[1], arr[2]], axis=1)
    raise ValueError(f"Unexpected embedding shape for combined modality: {arr.shape}")

def id2name_dict():
    return {v: k for k, v in LABELS_DICT.items()}

def canon_label(raw_label):
    if raw_label in LABELS_DICT:
        return raw_label

    inv = id2name_dict()
    if isinstance(raw_label, (int, np.integer)):
        return inv.get(int(raw_label), None)
    if isinstance(raw_label, str) and raw_label.isdigit():
        num = int(raw_label)
        return inv.get(num, None)
    if isinstance(raw_label, str) and raw_label in LABEL_MAP:
        return LABEL_MAP[raw_label]

    if isinstance(raw_label, str) and '|' in raw_label:
        name, idx = raw_label.split('|', 1)
        if name in LABEL_MAP:
            return LABEL_MAP[name]
        if name in LABELS_DICT:
            return name
        if idx.isdigit():
            num = int(idx)
            if num in inv:
                return inv[num]

    if raw_label in EVENT_TO_ID:
        cls_id = EVENT_TO_ID[raw_label]
        return inv.get(cls_id, None)

    return None

def filter_and_make_xy(emb, labels):
    emb = np.asarray(emb)
    N = emb.shape[0]
    std_names = [canon_label(l) for l in labels]

    indices = [i for i, name in enumerate(std_names) if (name is not None and i < N)]

    if len(labels) != N or len(indices) != N:
        logger.warning(f"[mismatch] len(labels)={len(labels)} vs emb.N={N} | kept={len(indices)}")
        left = max(0, N-3)
        right = min(len(labels), N+3)
        tail = labels[left:right]
        logger.warning(f"labels tail around N: {tail}")

    if len(indices) == 0:
        return np.empty((0, emb.shape[1])), np.array([]), indices

    X = emb[indices]
    y = np.array([LABELS_DICT[std_names[i]] for i in indices])
    return X, y, indices

def make_sample_weight(y_train: np.ndarray, K: int, use_weight: bool) -> np.ndarray | None:
    """balanced 클래스 가중치로 sample_weight 벡터 생성."""
    if not use_weight:
        return None
    classes = np.arange(K)
    cls_w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    wmap = {c: w for c, w in zip(classes, cls_w)}
    sw = np.array([wmap[y] for y in y_train], dtype=float)
    logger.info(f"class weights (balanced): {dict(zip(classes, cls_w))}")
    return sw

def train_and_eval(model_name, X_train, y_train, X_test, y_test, K, max_iter, sample_weight=None):
    """
    model_name: 'logistic' | 'xgb'
    반환: model, y_probs (N_test, K), class_report(dict), y_pred(np.ndarray)
    """
    target_names = [CLASS_LABELS[i] for i in range(K)]

    if model_name == "logistic":
        clf = LogisticRegression(
            max_iter=max_iter,
            n_jobs=-1,
            multi_class="auto",
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight)

        y_pred = clf.predict(X_test)
        y_probs = clf.predict_proba(X_test)

    elif model_name == "xgb":
        n_estimators = max(200, max_iter)  
        clf = XGBClassifier(
            objective="multi:softprob",
            num_class=K,
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=8,
            min_child_weight=1.0,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            reg_lambda=1.0,
            reg_alpha=0.0,
            eval_metric="mlogloss",
            n_jobs=-1,
            verbosity=1,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight)

        y_pred = clf.predict(X_test)
        y_probs = clf.predict_proba(X_test)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {acc:.4f}")

    report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    logger.info(f"y_pred dist: {collections.Counter(y_pred)}")
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(K))
    with np.printoptions(precision=4, suppress=True):
        logger.info(f"Confusion matrix (rows=true, cols=pred):\n{cm}")
        row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
        logger.info(f"Confusion matrix (row-normalized):\n{cm / row_sum}")

    return clf, y_probs, report, y_pred


def main(args):
    dataset_dir = args.dataset_dir if args.dataset_dir is not None else PATH_TO_PROCESSED_DATA

    path_to_output = resolve_output_path(dataset_dir, args.output_file)

    modality_type = args.modality_type
    model_name = args.model_name
    use_class_weight = (not args.no_class_weight)

    path_to_figures = os.path.join(path_to_output, "figures")
    path_to_models  = os.path.join(path_to_output, "models")
    path_to_probs   = os.path.join(path_to_output, "probs")
    os.makedirs(path_to_figures, exist_ok=True)
    os.makedirs(path_to_models,  exist_ok=True)
    os.makedirs(path_to_probs,   exist_ok=True)

    dataset_file        = "dataset.pickle"
    dataset_event_file  = "dataset_events_-1.pickle"
    test_emb_file       = "dataset_events_-1_test_emb.pickle"
    valid_emb_file      = "dataset_events_-1_valid_emb.pickle"
    train_emb_file      = "dataset_events_-1_train_emb.pickle"

    logger.info(f"modality_type: {modality_type}")
    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"path_to_output: {path_to_output}")
    logger.info(f"use_class_weight: {use_class_weight}")
    logger.info(f"dataset_file: {dataset_file}")
    logger.info(f"dataset_event_file: {dataset_event_file}")
    logger.info(f"test_emb_file: {test_emb_file}")
    logger.info(f"valid_emb_file: {valid_emb_file}")
    logger.info(f"train_emb_file: {train_emb_file}")

    path_to_dataset = os.path.join(dataset_dir, dataset_file)
    with open(path_to_dataset, "rb") as f:
        dataset = pickle.load(f)

    path_to_event_dataset = os.path.join(dataset_dir, dataset_event_file)
    with open(path_to_event_dataset, "rb") as f:
        dataset_events = pickle.load(f)

    path_to_eval_data = os.path.join(path_to_output, "eval_data")
    with open(os.path.join(path_to_eval_data, test_emb_file), "rb") as f:
        emb_test = pickle.load(f)
    with open(os.path.join(path_to_eval_data, valid_emb_file), "rb") as f:
        emb_valid = pickle.load(f)
    with open(os.path.join(path_to_eval_data, train_emb_file), "rb") as f:
        emb_train = pickle.load(f)

    path_to_label = {}
    for split, split_dataset in tqdm(dataset.items(), desc="Building path_to_label by split"):
        for patient_data in tqdm(split_dataset, leave=False, desc=f"{split} patients"):
            mrn = list(patient_data.keys())[0]
            for event, event_paths in patient_data[mrn].items():
                for event_path in event_paths:
                    path_to_label[event_path] = event

    labels_test  = np.array([path_to_label[event_path[0]] for event_path in dataset_events["test"]])
    labels_valid = np.array([path_to_label[event_path[0]] for event_path in dataset_events["valid"]])
    labels_train = np.array([path_to_label[event_path[0]] for event_path in dataset_events["train"]])

    logger.info(f"Counts (raw) - Train: {collections.Counter(labels_train)}, "
                f"Test: {collections.Counter(labels_test)}, Valid: {collections.Counter(labels_valid)}")
    logger.info(f"len(dataset_events['train']) = {len(dataset_events['train'])}")
    logger.info(f"len(labels_train) = {len(labels_train)}")

    if modality_type == "combined":
        emb_train = concat_modal_embeddings(emb_train)
        emb_valid = concat_modal_embeddings(emb_valid)
        emb_test  = concat_modal_embeddings(emb_test)
    else:
        target_index = MODALITY_TYPES.index(modality_type)
        emb_train = emb_train[target_index]
        emb_valid = emb_valid[target_index]
        emb_test  = emb_test[target_index]

    emb_train = np.asarray(emb_train)
    emb_valid = np.asarray(emb_valid)
    emb_test  = np.asarray(emb_test)

    logger.info(f"emb_train shape: {emb_train.shape}")
    logger.info(f"emb_valid shape: {emb_valid.shape}")
    logger.info(f"emb_test  shape: {emb_test.shape}")

    X_train, y_train, idx_tr = filter_and_make_xy(emb_train, labels_train)
    X_test,  y_test,  idx_te = filter_and_make_xy(emb_test,  labels_test)
    X_valid, y_valid, idx_va = filter_and_make_xy(emb_valid, labels_valid)

    logger.info(f"After filtering - Train: {X_train.shape}, Test: {X_test.shape}, Valid: {X_valid.shape}")
    logger.info(f"y_train dist (ids): {collections.Counter(y_train)}")
    logger.info(f"y_test  dist (ids): {collections.Counter(y_test)}")

    K = len(CLASS_LABELS)
    uniq = np.unique(y_train)
    logger.info(f"np.unique(y_train) = {uniq}")
    if not np.array_equal(uniq, np.arange(K)):
        logger.warning(f"Label indices are not 0..{K-1}. Check LABELS_DICT/CLASS_LABELS order.")
    logger.info(f"CLASS_LABELS (id->name): {list(CLASS_LABELS)}")
    logger.info(f"LABELS_DICT (name->id): {dict(LABELS_DICT)}")

    sample_weight = make_sample_weight(y_train, K, use_class_weight)

    model, y_probs, class_report, y_pred = train_and_eval(
        model_name, X_train, y_train, X_test, y_test, K, args.max_iter, sample_weight=sample_weight
    )

    path_to_models  = os.path.join(path_to_output, "models")
    path_to_probs   = os.path.join(path_to_output, "probs")
    os.makedirs(path_to_models, exist_ok=True)
    os.makedirs(path_to_probs,  exist_ok=True)

    logger.info(f"Saving model...")
    with open(os.path.join(path_to_models, f"{modality_type}_model_{model_name}.pickle"), 'wb') as file:
        pickle.dump(model, file)

    logger.info(f"Saving probabilities...")
    with open(os.path.join(path_to_probs, f"{modality_type}_y_probs_{model_name}.pickle"), 'wb') as file:
        pickle.dump(y_probs, file)

    logger.info(f"Saving class report...")
    with open(os.path.join(path_to_probs, f"{modality_type}_class_report_{model_name}.pickle"), 'wb') as file:
        pickle.dump(class_report, file)

    logger.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with class weights and evaluate.")
    parser.add_argument("--output_file", type=str, required=True, help="Output directory name or absolute path")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to preprocessed data (where dataset.pickle lives)")
    parser.add_argument("--modality_type", type=str, choices=["respiratory", "sleep_stages", "ekg", "combined"], default="combined")
    parser.add_argument("--model_name", type=str, default="logistic", choices=["logistic", "xgb"], help="Model type")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iter (LR) / n_estimators fallback (XGB)")
    parser.add_argument("--no_class_weight", action="store_true", help="Disable class-weight balancing")
    args = parser.parse_args()
    main(args)
