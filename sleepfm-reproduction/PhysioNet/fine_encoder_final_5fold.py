import sys
sys.path.append('/ssd/kdpark/sleepfm-codebase')

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score)
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision 

from sleepfm.model import models
from config import CHANNEL_DATA_IDS, LABELS_DICT

# 수정 Dataset
class EventDatasetSupervised(torchvision.datasets.VisionDataset):
    def __init__(self, root=None, split="train", modality_type="sleep_stages", samples=None):
        self.split = split
        self.modality_type = modality_type

        if samples is not None:
            self.dataset = samples
        else:
            with open(root, "rb") as f:
                self.dataset = pickle.load(f)
            if split == "combined":
                self.dataset = self.dataset["pretrain"] + self.dataset["train"]
            else:
                self.dataset = self.dataset[split]
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_path, event = self.dataset[index]
        event_num = LABELS_DICT[event]
        data = np.load(data_path).astype(np.float32)

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
        
        return data, event_num


BATCH_SIZE = 32
MODALITY = "combined"
EPOCHS = 10
LEARNING_RATE = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(LABELS_DICT)


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

    ckpt = torch.load("/ssd/kdpark/sleepfm-codebase/physionet_final/my_run_final/best.pt", map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    encoder.load_state_dict(state_dict, strict=False)
    for p in encoder.parameters():
        p.requires_grad = True
    return encoder.to(device)


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
        auroc_mean, ci_low, ci_high = compute_bootstrap_ci((y_test == i).astype(int), y_probs[:, i])
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

# 5-Fold
def main():
    dataset_file = "/ssd/kdpark/sleepfm-codebase/physionet_final/dataset_events_-1.pickle"
    with open(dataset_file, 'rb') as f:
        data = pickle.load(f)

    all_data = data['train']
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(all_data)):
        print(f"\n Fold {fold_id + 1}/5")
        train_split = [all_data[i] for i in train_idx]
        test_split = [all_data[i] for i in test_idx]

        train_dataset = EventDatasetSupervised(samples=train_split, modality_type=MODALITY)
        test_dataset = EventDatasetSupervised(samples=test_split, modality_type=MODALITY)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        encoder = load_encoder(MODALITY)
        classifier_head = nn.Linear(512, NUM_CLASSES).to(device)
        full_model = nn.Sequential(encoder, classifier_head)

        optimizer = torch.optim.Adam(full_model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        print("인코더 파인튜닝")
        for epoch in range(EPOCHS):
            loss = train_encoder(full_model, train_loader, optimizer, criterion)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

        print("임베딩 추출")
        feature_encoder = nn.Sequential(encoder).to(device)
        X_train, y_train = extract_embeddings(feature_encoder, train_loader)
        X_test, y_test = extract_embeddings(feature_encoder, test_loader)

        print("로지스틱 회귀 학습 및 평가")
        class_labels = list(LABELS_DICT.keys())
        train_logistic_regression(X_train, X_test, y_train, y_test, class_labels)

if __name__ == "__main__":
    main()
