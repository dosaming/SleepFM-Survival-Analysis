import sys
sys.path.append('/ssd/kdpark/sleepfm-codebase')

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 디스크립터 이슈 완화
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from sleepfm.model.dataset import EventDatasetSupervised
from sleepfm.model import models
from config_shhs_new import CHANNEL_DATA_IDS, LABELS_DICT

# 설정
BATCH_SIZE = 128
MODALITY = "sleep_stages"
EPOCHS = 5
LEARNING_RATE = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(LABELS_DICT)

# 인코더 체크포인트 불러오고 freeze 해제
def load_encoder(modality):
    key_map = {
        "sleep_stages": "Sleep_Stages",
        "respiratory": "Respiratory",
        "ekg": "EKG",
        "combined": ["Respiratory", "Sleep_Stages", "EKG"]
    }
    key = key_map[modality]
    in_channel = sum(len(CHANNEL_DATA_IDS[k]) for k in key) if isinstance(key, list) else len(CHANNEL_DATA_IDS[key])

    encoder = models.EffNet(in_channel=in_channel, stride=2, dilation=1)
    encoder.fc = nn.Linear(encoder.fc.in_features, 512)

    ckpt = torch.load("/ssd/kdpark/sleepfm-codebase/outputs_shhs_list/best.pt", map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    encoder.load_state_dict(state_dict, strict=False)

    for p in encoder.parameters():
        p.requires_grad = True

    return encoder.to(device)

# 임베딩 추출
def extract_embeddings(model, loader):
    model.eval()
    X_all, y_all = [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="임베딩 추출"):
            X = X.to(device)
            emb = model(X)
            X_all.append(emb.cpu().numpy())
            y_all.append(y.cpu().numpy())
    return np.concatenate(X_all), np.concatenate(y_all)

# 인코더 파인튜닝 학습
def train_encoder(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X, y in tqdm(loader, desc="인코더 파인튜닝"):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)

# AUROC 95% CI 계산
def compute_bootstrap_ci(y_true, y_probs, n_bootstraps=1000, alpha=0.95):
    aurocs = []
    rng = np.random.RandomState(seed=42)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_probs[indices])
        aurocs.append(score)

    sorted_scores = np.sort(aurocs)
    lower = np.percentile(sorted_scores, ((1 - alpha) / 2) * 100)
    upper = np.percentile(sorted_scores, (1 - (1 - alpha) / 2) * 100)
    return np.mean(aurocs), lower, upper

# 로지스틱 회귀
def train_logistic_regression(X_train, X_test, y_train, y_test, class_labels):
    print("로지스틱 회귀 학습")
    model = LogisticRegression(class_weight="balanced", max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n Accuracy: {acc:.4f}\n")
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10} {'AUROC (95% CI)':<20}")
    for i, cls in enumerate(class_labels):
        precision = report[cls]['precision']
        recall = report[cls]['recall']
        f1 = report[cls]['f1-score']
        support = int(report[cls]['support'])

        auroc_mean, ci_low, ci_high = compute_bootstrap_ci(
            (y_test == i).astype(int), y_probs[:, i]
        )

        print(f"{cls:<10} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10} {auroc_mean:.3f} ({ci_low:.3f}, {ci_high:.3f})")

    print(f"\n{'Metric Type':<13} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUROC':<10}")
    for avg in ["macro avg", "weighted avg"]:
        precision = report[avg]["precision"]
        recall = report[avg]["recall"]
        f1 = report[avg]["f1-score"]
        print(f"{avg.title():<13} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {'-':<10}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def main():
    print("데이터셋 로드중")
    dataset_file = "/ssd/kdpark/sleepfm-codebase/shhs_segments_125/dataset_events_-1.pickle"  
    train_dataset = EventDatasetSupervised(root=dataset_file, split="train", modality_type=MODALITY)
    test_dataset  = EventDatasetSupervised(root=dataset_file, split="test",  modality_type=MODALITY)

    # 학습용 로더(기존대로)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("인코더 로드")
    encoder = load_encoder(MODALITY)

    classifier_head = nn.Linear(512, NUM_CLASSES).to(device)
    full_model = nn.Sequential(encoder, classifier_head)

    optimizer = torch.optim.Adam(full_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("인코더 파인튜닝")
    for epoch in range(EPOCHS):
        loss = train_encoder(full_model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

    # ========= 학습 직후 체크포인트 저장 (임베딩 추출 전에!) =========
    out_root = "/ssd/kdpark/sleepfm-codebase/outputs_shhs_list/runs"
    os.makedirs(out_root, exist_ok=True)
    run_name = time.strftime("%y%m%d_%H%M%S") + f"_{MODALITY}"
    save_dir = os.path.join(out_root, run_name)
    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = os.path.join(save_dir, "ft_last.pt")
    torch.save({
        "epoch": EPOCHS,
        "state_dict_encoder": encoder.state_dict(),
        "state_dict_full": full_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            "modality": MODALITY,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_classes": NUM_CLASSES,
            "dataset_file": dataset_file
        }
    }, ckpt_path)
    print(f"[SAVE] checkpoint -> {ckpt_path}")
    # ===========================================================

    # 임베딩용 로더: 워커 0으로 별도 생성 (FD 고갈 방지)
    embed_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=0, pin_memory=False)
    embed_test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=0, pin_memory=False)

    print("인코더에서 임베딩 추출")
    encoder.eval()
    feature_encoder = nn.Sequential(encoder).to(device)
    X_train, y_train = extract_embeddings(feature_encoder, embed_train_loader)
    X_test,  y_test  = extract_embeddings(feature_encoder,  embed_test_loader)

    print("로지스틱 회귀 학습")
    class_labels = list(LABELS_DICT.keys())
    train_logistic_regression(X_train, X_test, y_train, y_test, class_labels)

if __name__ == "__main__":
    main()
