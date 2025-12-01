

"""
라벨:
  Wake=0, N1=1, N2=2, N3=3, REM=4
  Stage 4 (있으면) -> N3(=3)로 
"""

import os
import gzip
import pickle
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import mne
import xml.etree.ElementTree as ET
import scipy.signal
from tqdm import tqdm
from multiprocessing import Pool

import config_shhs as config
from config_shhs import ALL_CHANNELS, PATH_TO_PROCESSED_DATA
try:
    from config_shhs import OPTIONAL_CHANNEL_RENAMES
except Exception:
    OPTIONAL_CHANNEL_RENAMES = {}

EPOCH_SEC_SIZE_DEFAULT = 30

LABEL_MAP = {
    "Sleep stage W": "Wake",
    "Sleep stage N1": "Stage 1",
    "Sleep stage N2": "Stage 2",
    "Sleep stage N3": "Stage 3",
    "Sleep stage R": "REM",
    "W": "Wake",
    "N1": "Stage 1",
    "N2": "Stage 2",
    "N3": "Stage 3",
    "REM": "REM",
    "wake": "Wake",
    "nonrem1": "Stage 1",
    "nonrem2": "Stage 2",
    "nonrem3": "Stage 3",
    "rem": "REM",

    "Wake|0": "Wake",
    "Stage 1 sleep|1": "Stage 1",
    "Stage 2 sleep|2": "Stage 2",
    "Stage 3 sleep|3": "Stage 3",
    "Stage 4 sleep|4": "Stage 3",  # N3로 합치기
    "REM sleep|5": "REM",
}
EVENT_TO_ID_FINAL = {"Wake": 0, "Stage 1": 1, "Stage 2": 2, "Stage 3": 3, "REM": 4}


def normalize_label(desc: str) -> str:
    if desc in LABEL_MAP:
        return LABEL_MAP[desc]
    if "|" in desc:
        base = desc.split("|")[0].strip()
        base = base.replace(" sleep", "")
        if base in LABEL_MAP:
            return LABEL_MAP[base]
    key = desc.strip()
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    key2 = key.lower()
    return LABEL_MAP.get(key2, "")


def label_to_id(desc: str) -> int:
    std = normalize_label(desc)
    return EVENT_TO_ID_FINAL.get(std, -1)

def _stem_key_from_name(name: str) -> str:

    n = name.lower()
    if n.endswith(".xml.gz"):
        base = n[:-7]
    elif n.endswith(".xml"):
        base = n[:-4]
    elif n.endswith(".edf"):
        base = n[:-4]
    else:
        base = n
    base = base.replace("-nsrr", "").replace("-profusion", "")
    return base


def find_shhs_pairs(edf_root: str, xml_root: str) -> List[Tuple[str, str]]:
    edf_root = Path(edf_root)
    xml_root = Path(xml_root)

    edfs = [p for p in edf_root.rglob("*") if p.is_file() and p.suffix.lower() == ".edf"]

    xml_candidates = []
    for p in xml_root.rglob("*"):
        if not p.is_file():
            continue
        n = p.name.lower()
        if n.endswith(".xml") or n.endswith(".xml.gz"):
            xml_candidates.append(p)

    xml_map = {_stem_key_from_name(p.name): p for p in xml_candidates}

    pairs = []
    for e in edfs:
        key = _stem_key_from_name(e.name)
        x = xml_map.get(key)
        if x is not None:
            pairs.append((str(e), str(x)))

    if len(pairs) == 0:
        print(f"[DEBUG] edfs found: {len(edfs)}, xmls found: {len(xml_candidates)}")
        for smp in edfs[:5]:
            print("[EDF]", smp.name, "->", _stem_key_from_name(smp.name))
        for smp in xml_candidates[:5]:
            print("[XML]", smp.name, "->", _stem_key_from_name(smp.name))

    return pairs

def parse_shhs_annotations(xml_path: str, epoch_sec: int = 30):

    p = Path(xml_path)
    if p.name.lower().endswith(".xml.gz"):
        with gzip.open(p, "rb") as f:
            tree = ET.parse(f)
    else:
        tree = ET.parse(p)
    root = tree.getroot()

    out = []

    for se in root.findall(".//ScoredEvent"):
        etype = (se.findtext("EventType") or "").strip()
        econcept = (se.findtext("EventConcept") or "").strip()
        start = float(se.findtext("Start") or 0.0)
        duration = float(se.findtext("Duration") or 0.0)
        if ("stage" in etype.lower()) or ("sleep" in etype.lower()):
            out.append((start, duration, econcept))

    if out:
        out.sort(key=lambda x: x[0])
        return out

    stages_node = root.find(".//SleepStages")
    if stages_node is not None:
        labels = []
        for node in stages_node.findall(".//SleepStage"):
            txt = (node.text or "").strip()
            if not txt:
                continue
            try:
                lbl = int(txt)
            except ValueError:
                continue
            labels.append(lbl)

        if labels:
            def num2desc(n: int) -> str:
                if n == 0: return "Wake|0"
                if n == 1: return "Stage 1 sleep|1"
                if n == 2: return "Stage 2 sleep|2"
                if n == 3: return "Stage 3 sleep|3"
                if n == 4: return "Stage 4 sleep|4"
                if n == 5: return "REM sleep|5"
                return f"Unknown|{n}"

            out = []
            for i, n in enumerate(labels):
                start = i * float(epoch_sec)
                duration = float(epoch_sec)
                out.append((start, duration, num2desc(n)))
            return out

    return []

def apply_channel_renames(raw: mne.io.BaseRaw):
    """config_shhs.OPTIONAL_CHANNEL_RENAMES가 있으면 채널명 보정."""
    if not OPTIONAL_CHANNEL_RENAMES:
        return
    rename_map = {orig: OPTIONAL_CHANNEL_RENAMES.get(orig, orig) for orig in raw.ch_names}
    raw.rename_channels(rename_map)

def process_one_record(
    edf_path: str,
    xml_path: str,
    out_root: str,
    target_fs: int = 256,
    epoch_sec: int = EPOCH_SEC_SIZE_DEFAULT,
) -> int:

    record_id = Path(edf_path).stem
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    apply_channel_renames(raw)

    triples = parse_shhs_annotations(xml_path, epoch_sec=epoch_sec)
    if not triples:
        print(f"[SKIP no annotations] {Path(xml_path).name}")
        return 0

    onsets   = [t[0] for t in triples]
    durations= [t[1] for t in triples]
    descs    = [t[2] for t in triples]

    ann = mne.Annotations(onset=onsets, duration=durations, description=descs)
    raw.set_annotations(ann)

    events, event_id = mne.events_from_annotations(raw, chunk_duration=float(epoch_sec))
    if events is None or len(events) == 0:
        print(f"[SKIP no events] {Path(edf_path).name}")
        return 0

    sfreq = raw.info["sfreq"]
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0.0,
        tmax=epoch_sec - 1.0 / sfreq,
        baseline=None,
        preload=True,
        event_repeated="drop",
    )
    epochs.drop_bad()
    if epochs is None or len(epochs) == 0:
        print(f"[SKIP empty epochs] {Path(edf_path).name}")
        return 0

    ch_names = epochs.ch_names
    missing = [ch for ch in ALL_CHANNELS if ch not in ch_names]
    if missing:
        preview = ", ".join(ch_names[:8])
        print(f"[SKIP missing channels] {Path(edf_path).name}: missing={missing} | seen[:8]=[{preview}]")
        return 0

    indices = [ch_names.index(ch) for ch in ALL_CHANNELS]

    x_dir = Path(out_root) / "X" / record_id
    y_path = Path(out_root) / "Y" / f"{record_id}.pickle"
    x_dir.mkdir(parents=True, exist_ok=True)
    y_path.parent.mkdir(parents=True, exist_ok=True)

    labels = {}
    inv_event_id = {v: k for k, v in epochs.event_id.items()}  
    num_target = int(epoch_sec * target_fs)

    saved = 0
    for i in range(len(epochs)):
        ep = epochs[i].get_data()[0]      
        data = ep[indices, :]             
        if data.shape[1] == 0:
            continue

        resampled = scipy.signal.resample(data, num_target, axis=1).astype(np.float32)

        code = epochs.events[i, -1]
        desc = inv_event_id.get(code, "")
        y = label_to_id(desc)
        if y < 0:
            print(f"[SKIP label map] {record_id} epoch {i}: desc='{desc}'")
            continue

        fname = f"{record_id}_{i}.npy"
        np.save(x_dir / fname, resampled)
        labels[fname] = int(y)
        saved += 1

    if saved > 0:
        with open(y_path, "wb") as f:
            pickle.dump(labels, f)
    else:
        print(f"[NOTE no epochs saved] {record_id} (likely missing ALL_CHANNELS or label mapping issues)")

    return saved

def process_one_record_star(args_tuple) -> int:
    """(edf_path, xml_path, out_root, target_fs, epoch_sec) 튜플을 받아 언팩 실행."""
    return process_one_record(*args_tuple)

def parse_args():
    p = argparse.ArgumentParser(description="SHHS -> SleepFM format extractor (with debug)")
    p.add_argument("--shhs_edf_dir", type=str, required=True,
                   help='SHHS EDF root (e.g., ".../polysomnography/edfs/shhs1")')
    p.add_argument("--shhs_xml_dir", type=str, required=True,
                   help='SHHS XML root (e.g., ".../annotations-events-profusion/shhs1" or ".../annotations-events-nsrr")')
    p.add_argument("--target_sampling_rate", type=int, default=256,
                   help="Target sampling rate (default: 256 Hz)")
    p.add_argument("--num_threads", type=int, default=8, help="Parallel workers")
    p.add_argument("--epoch_sec", type=int, default=EPOCH_SEC_SIZE_DEFAULT)
    return p.parse_args()

def main():
    args = parse_args()

    print(f"[INFO] PATH_TO_PROCESSED_DATA = {PATH_TO_PROCESSED_DATA}")

    pairs = find_shhs_pairs(args.shhs_edf_dir, args.shhs_xml_dir)
    print(f"[INFO] Found {len(pairs)} EDF-XML pairs")

    out_root = PATH_TO_PROCESSED_DATA
    os.makedirs(out_root, exist_ok=True)

    if len(pairs) == 0:
        print("[HINT] Check --shhs_edf_dir / --shhs_xml_dir paths and filename suffixes (-nsrr/-profusion, .xml.gz).")
        print(f"[DONE] Saved epochs: 0  |  Records processed: 0  -> {out_root}")
        return

    args_list = [(edf, xml, out_root, args.target_sampling_rate, args.epoch_sec)
                 for edf, xml in pairs]

    saved_total = 0
    with Pool(args.num_threads) as pool:
        for saved in tqdm(pool.imap_unordered(process_one_record_star, args_list), total=len(args_list)):
            saved_total += saved

    print(f"[DONE] Saved epochs: {saved_total}  |  Records processed: {len(pairs)}  -> {out_root}")

if __name__ == "__main__":
    main()
