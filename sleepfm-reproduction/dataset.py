import torch
import cv2
import os
import numpy as np
import pandas
import pickle
import torchvision
import random
import math
import time
from typing import Any 
from config_shhs_new import (CHANNEL_DATA, ALL_CHANNELS, CHANNEL_DATA_IDS,EVENT_TO_ID, LABELS_DICT)

from loguru import logger
import shutil
import sys
sys.path.append("../")


import re  # ✅ 추가: 라벨 문자열 파싱용

# ✅ 라벨을 어떤 형식으로 받아도 정수 ID로 변환하는 헬퍼
def _to_label_id(event, labels_dict=None, event_to_id=None) -> int:
    """
    event: int, np.int, '2', 'N2', 'REM', 'Stage 2 sleep|2' 등
    우선순위: (이미 정수) → (EVENT_TO_ID 매핑) → (LABELS_DICT 매핑) → (숫자 문자열) → (문자열 끝의 숫자)
    """
    # 1) 이미 정수형
    if isinstance(event, (int, np.integer)):
        return int(event)

    # 2) 문자열 처리
    if isinstance(event, str):
        s = event.strip()

        # 2-1) 표준 이벤트명 매핑이 있으면 먼저 사용 (권장)
        if event_to_id is not None and s in event_to_id:
            return int(event_to_id[s])

        # 2-2) LABELS_DICT가 문자열 키면 여기서도 시도
        if labels_dict is not None and s in labels_dict:
            return int(labels_dict[s])

        # 2-3) 순수 숫자 문자열
        if s.isdigit():
            return int(s)

        # 2-4) '...|2' 또는 문자열 끝 숫자 패턴
        m = re.search(r'(\d+)\s*$', s)
        if m:
            return int(m.group(1))

    # 3) 마지막 시도: LABELS_DICT가 id->name 형태인 경우 대비(옵션)
    try:
        if labels_dict is not None:
            inv = {v: k for k, v in labels_dict.items()}
            if event in inv:
                return int(event)
    except Exception:
        pass

    raise KeyError(f"Unknown label format: {event!r}")

# 기존 다른 코드 전부 이 클래스로
class EventDatasetSupervised(torchvision.datasets.VisionDataset):
    def __init__(self, root, split="train", modality_type="sleep_stages"):
        start = time.time()
        self.split = split
        self.modality_type = modality_type

        with open(root, "rb") as f:
            self.dataset = pickle.load(f)

        if split == "combined":
            # 필요 시 존재 여부 확인 권장: 'pretrain' 키가 없으면 KeyError
            self.dataset = self.dataset["pretrain"] + self.dataset["train"]
        else:
            self.dataset = self.dataset[split]
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_path = self.dataset[index][0]
        event     = self.dataset[index][1]

        # ✅ 안전한 라벨 변환
        event_num = _to_label_id(event, labels_dict=LABELS_DICT, event_to_id=EVENT_TO_ID)

        # ✅ 파일 핸들 즉시 닫히도록 컨텍스트 매니저 사용
        with open(data_path, "rb") as f:
            data = np.load(f, allow_pickle=False).astype(np.float32, copy=False)

        if self.modality_type == "respiratory":
            data = data[CHANNEL_DATA_IDS["Respiratory"]]
        elif self.modality_type == "sleep_stages":
            data = data[CHANNEL_DATA_IDS["Sleep_Stages"]]
        elif self.modality_type == "ekg":
            data = data[CHANNEL_DATA_IDS["EKG"]]
        elif self.modality_type == "combined":
            all_ids = CHANNEL_DATA_IDS["Respiratory"] + CHANNEL_DATA_IDS["Sleep_Stages"] + CHANNEL_DATA_IDS["EKG"]
            data = data[all_ids]
        else:
            raise ValueError(f'Modality type "{self.modality_type}" is not recognized.')
        
        # ✅ nsrrid 추출 (예: shhs1-200001/... → "200001")
        parent = os.path.basename(os.path.dirname(data_path))
        fname  = os.path.basename(data_path)
        m = re.search(r"shhs\d+-(\d{5,7})", parent)
        if m:
            nsrrid = m.group(1)
        else:
            m = re.search(r"(\d{5,7})", fname)
            nsrrid = m.group(1) if m else None
        meta = {"nsrrid": nsrrid}

        return data, event_num, meta   # ✅ meta까지 반환


class EventDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, split="train", modality_type=["respiratory", "sleep_stages", "ekg"]):
        start = time.time()
        self.split = split
        if isinstance(modality_type, list):
            self.modality_type = modality_type
        else:
            self.modality_type = [modality_type]

        with open(root, "rb") as f:
            self.dataset = pickle.load(f)

        self.dataset = self.dataset[split]
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_path = self.dataset[index][0]

        # 컨텍스트 매니저로 즉시 닫기
        with open(data_path, "rb") as f:
            data = np.load(f, allow_pickle=False).astype(np.float32, copy=False)
        
        target: Any = []
        for t in self.modality_type:
            if t == "respiratory":
                resp_data = data[CHANNEL_DATA_IDS["Respiratory"]]
                target.append(resp_data)
            elif t == "sleep_stages":
                sleep_data = data[CHANNEL_DATA_IDS["Sleep_Stages"]]
                target.append(sleep_data)
            elif t == "ekg":
                ekg_data = data[CHANNEL_DATA_IDS["EKG"]]
                target.append(ekg_data)
            else:
                raise ValueError(f'Target type "{t}" is not recognized.')
        
        return target


_cache = {}
def cache_csv(path, sep=None):
    if path in _cache:
        return _cache[path]
    else:
        x = pandas.read_csv(path, sep=sep)
        _cache[path] = x
        return x

_cache = {}
def cache_pkl(path):
    if path in _cache:
        return _cache[path]
    else:
        with open(path, "rb") as f:
            x = pickle.load(f)
        _cache[path] = x
        return x
