#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/ssd/kdpark/sleepfm-codebase')

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

import os
import argparse
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sleepfm.model.dataset import EventDatasetSupervised
from sleepfm.model import models
from config_shhs_new import CHANNEL_DATA_IDS, LABELS_DICT


def set_mp_file_system():
    import torch.multiprocessing as mp
    try:
        mp.set_sharing_strategy('file_system')
    except Exception:
        pass


def class_order_from_labels_dict(labels_dict):
    items = sorted(labels_dict.items(), key=lambda kv: kv[1])
    return [k for k, _ in items]


def load_encoder_for_modality(modality, device):
    key_map = {
        "sleep_stages": "Sleep_Stages",
        "respiratory": "Respiratory",
        "ekg": "EKG",
        "combined": ["Respiratory", "Sleep_Stages", "EKG"],
    }
    key = key_map[modality]
    in_channel = (
        sum(len(CHANNEL_DATA_IDS[k]) for k in key)
        if isinstance(key, list) else len(CHANNEL_DATA_IDS[key])
    )

    encoder = models.EffNet(in_channel=in_channel, stride=2, dilation=1)
    # 512-d 임베딩 헤드
    encoder.fc = nn.Linear(encoder.fc.in_features, 512)
    return encoder.to(device)


def load_checkpoint_into_encoder(encoder, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict_encoder" in ckpt:
        sd = ckpt["state_dict_encoder"]
    else:
        full_sd = ckpt.get("state_dict_full", {})
        sd = {k.replace("0.", ""): v for k, v in full_sd.items() if k.startswith("0.")}
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing:
        print("[WARN] missing keys:", missing)
    if unexpected:
        print("[WARN] unexpected keys:", unexpected)


@torch.no_grad()
def extract_embeddings(model, loader, device, expect_meta=True):
    """
    반환:
      - Z: (N, 512) float32  (epoch/segment 레벨)
      - y: (N,)               (있으면)
      - ids: (N,) object      (subject_id; 없으면 None)
    """
    model.eval()
    Z_list, y_list, id_list = [], [], []

    for batch in tqdm(loader, desc="임베딩 추출"):

        X = batch[0].to(device, non_blocking=True)
        z = model(X)  # [B, 512]
        Z_list.append(z.cpu().numpy())

        # y
        if len(batch) >= 2 and isinstance(batch[1], torch.Tensor) and batch[1].dim() == 1:
            y_list.append(batch[1].cpu().numpy())

        # subject_id 탐색
        subj_ids = None
        if expect_meta:
            meta = None
            if len(batch) >= 3 and isinstance(batch[2], dict):
                meta = batch[2]
            elif len(batch) >= 2 and isinstance(batch[1], dict):
                meta = batch[1]

            if meta is not None:
                for k in ["nsrrid", "subject_id", "record_id", "subject", "rec_id", "pid", "id"]:
                    if k in meta:
                        v = meta[k]
                        if isinstance(v, torch.Tensor):
                            subj_ids = v.cpu().tolist()
                        elif isinstance(v, np.ndarray):
                            subj_ids = v.tolist()
                        elif isinstance(v, (list, tuple)):
                            subj_ids = list(v)
                        else:
                            subj_ids = [v] * X.shape[0]
                        break

        if subj_ids is None:
            id_list.append([None] * X.shape[0])
        else:
            id_list.append(subj_ids)

    Z = np.concatenate(Z_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0) if len(y_list) > 0 else None
    ids = sum(id_list, [])
    ids = np.array(ids, dtype=object) if any(v is not None for v in ids) else None
    return Z, y, ids


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[SAVE] {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings -> save embeddings_{split}.pickle (segment-level only)"
    )
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "valid", "test"])
    parser.add_argument("--modality", type=str, default="sleep_stages",
                        choices=["sleep_stages", "respiratory", "ekg", "combined"])
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    set_mp_file_system()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Dataset & Loader
    ds = EventDatasetSupervised(
        root=args.dataset_file,
        split=args.split,
        modality_type=args.modality
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # 2) Encoder & Checkpoint
    encoder = load_encoder_for_modality(args.modality, device)
    load_checkpoint_into_encoder(encoder, args.ckpt_path)
    feature_encoder = nn.Sequential(encoder).to(device)

    # 3) Embeddings (epoch/segment 레벨)
    Z, y, ids = extract_embeddings(feature_encoder, loader, device, expect_meta=True)
    print(f"[INFO] embeddings: {Z.shape}, labels: {None if y is None else y.shape}, ids: {'ok' if ids is not None else 'None'}")

    os.makedirs(args.out_dir, exist_ok=True)
    class_labels = class_order_from_labels_dict(LABELS_DICT)

    # 4) 저장: embeddings_{split}.pickle
    emb_path = os.path.join(args.out_dir, f"embeddings_{args.split}.pickle")
    save_pickle(
        {
            "X": Z,           # (N, 512)
            "y": y,           # (N,) or None
            "ids": ids,       # (N,) or None (subject_id)
            "labels": class_labels,
            "meta": {
                "dataset_file": args.dataset_file,
                "split": args.split,
                "modality": args.modality,
                "ckpt": args.ckpt_path,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        },
        emb_path,
    )

    print("\nDone")
    print(f"- embeddings saved to: {emb_path}")


if __name__ == "__main__":
    main()
