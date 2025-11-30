import os
import sys
import math
import pickle
import time
import datetime

import click
import torch
import tqdm
import numpy as np
from loguru import logger

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.append("/ssd/kdpark/sleepfm-codebase")

# 내부 모듈 import
from config import (
    PATH_TO_PROCESSED_DATA,
    CHANNEL_DATA_IDS,
    EMBED_SAVE_PATH,
)
from sleepfm.model import models
from sleepfm.model.dataset import EventDataset as Dataset


@click.command("generate_eval_embed")
@click.argument("run_name", type=str)  # 예: my_run_final
@click.option(
    "--data_dir",
    type=str,
    default=None,
    help="경로: dataset_events_-1.pickle 이 있는 디렉토리 (기본: PATH_TO_PROCESSED_DATA)",
)
@click.option("--dataset_file", type=str, default="dataset_events_-1.pickle")
@click.option("--batch_size", type=int, default=32)
@click.option("--num_workers", type=int, default=2)
@click.option(
    "--splits",
    type=str,
    default="train,valid,test",
    help="사용할 데이터 split 리스트. 예: 'train,valid,test' 또는 'test'",
)
def generate_eval_embed(
    run_name,
    data_dir,
    dataset_file,
    batch_size,
    num_workers,
    splits,
):
    """
    RUN_NAME: outputs/RUN_NAME/ 안의 best.pt를 사용해
    데이터셋에서 임베딩을 추출하고
    outputs/RUN_NAME/eval_data/ 에 *_emb.pickle 저장.
    """

    # -----------------------
    # 경로 설정
    # -----------------------
    if data_dir is None:
        data_dir = PATH_TO_PROCESSED_DATA

    # EMBED_SAVE_PATH에서 outputs 루트 추출
    # "/ssd/.../outputs/my_run/embeddings" -> "/ssd/.../outputs"
    outputs_root = os.path.dirname(os.path.dirname(EMBED_SAVE_PATH))

    # 이 run의 체크포인트/출력 디렉토리
    output_dir = os.path.join(outputs_root, run_name)

    logger.info(f"Data dir       : {data_dir}")
    logger.info(f"Dataset file   : {dataset_file}")
    logger.info(f"Outputs root   : {outputs_root}")
    logger.info(f"Run directory  : {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # "train,valid,test" -> ["train","valid","test"]
    splits = [s.strip() for s in splits.split(",") if s.strip()]

    # -----------------------
    # Dataset 로드
    # -----------------------
    path_to_dataset_pickle = os.path.join(data_dir, dataset_file)

    dataset = {
        split: Dataset(
            path_to_dataset_pickle,
            split=split,
            modality_type=["respiratory", "sleep_stages", "ekg"],
        )
        for split in splits
    }

    # -----------------------
    # 모델 정의 (config 기반 채널 수 사용)
    # -----------------------

    # Respiratory
    in_channel_resp = len(CHANNEL_DATA_IDS["Respiratory"])
    model_resp = models.EffNet(in_channel=in_channel_resp, stride=2, dilation=1)
    model_resp.fc = torch.nn.Linear(model_resp.fc.in_features, 512)

    # Sleep_Stages
    in_channel_sleep = len(CHANNEL_DATA_IDS["Sleep_Stages"])  # checkpoint 기준 5채널
    model_sleep = models.EffNet(in_channel=in_channel_sleep, stride=2, dilation=1)
    model_sleep.fc = torch.nn.Linear(model_sleep.fc.in_features, 512)

    # EKG
    in_channel_ekg = len(CHANNEL_DATA_IDS["EKG"])
    model_ekg = models.EffNet(in_channel=in_channel_ekg, stride=2, dilation=1)
    model_ekg.fc = torch.nn.Linear(model_ekg.fc.in_features, 512)

    if device.type == "cuda":
        model_resp = torch.nn.DataParallel(model_resp)
        model_sleep = torch.nn.DataParallel(model_sleep)
        model_ekg = torch.nn.DataParallel(model_ekg)

    model_resp.to(device)
    model_sleep.to(device)
    model_ekg.to(device)

    # -----------------------
    # 체크포인트 로드
    # -----------------------
    ckpt_path = os.path.join(output_dir, "best.pt")
    logger.info(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

    temperature = checkpoint.get("temperature", None)
    if temperature is not None:
        logger.info(f"Loaded temperature: {temperature}")

    # ✅ checkpoint 키 이름: resp_state_dict / sleep_state_dict / ekg_state_dict
    model_resp.load_state_dict(checkpoint["resp_state_dict"])
    model_sleep.load_state_dict(checkpoint["sleep_state_dict"])
    model_ekg.load_state_dict(checkpoint["ekg_state_dict"])

    model_resp.eval()
    model_sleep.eval()
    model_ekg.eval()

    # -----------------------
    # 임베딩 저장 경로
    # -----------------------
    path_to_save = os.path.join(output_dir, "eval_data")
    os.makedirs(path_to_save, exist_ok=True)
    logger.info(f"Embeddings will be saved to: {path_to_save}")

    # -----------------------
    # 각 split별로 임베딩 추출
    # -----------------------
    for split in splits:
        logger.info(f"Processing split: {split}")
        dataloader = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )

        emb = [[], [], []]  # resp, sleep, ekg

        with torch.no_grad():
            with tqdm.tqdm(total=len(dataloader), desc=f"Embeddings for {split}") as pbar:
                for (resp, sleep, ekg) in dataloader:
                    resp = resp.to(device, dtype=torch.float)
                    sleep = sleep.to(device, dtype=torch.float)
                    ekg = ekg.to(device, dtype=torch.float)

                    # 🔥 Sleep_Stages 채널 mismatch 처리
                    # checkpoint 기준 in_channel_sleep(=5)인데
                    # 실제 데이터가 [B,4,T]로 들어오는 경우 dummy 채널을 뒤에 붙여서 [B,5,T]로 맞춘다.
                    if sleep.dim() == 3 and sleep.size(1) != in_channel_sleep:
                        logger.warning(
                            f"Sleep channels mismatch: got {sleep.size(1)}, "
                            f"expected {in_channel_sleep}. Padding with zeros."
                        )
                        if sleep.size(1) < in_channel_sleep:
                            pad_channels = in_channel_sleep - sleep.size(1)
                            zeros = torch.zeros(
                                sleep.size(0),
                                pad_channels,
                                sleep.size(2),
                                device=sleep.device,
                                dtype=sleep.dtype,
                            )
                            sleep = torch.cat([sleep, zeros], dim=1)
                        else:
                            # 혹시 채널이 더 많으면 잘라서 맞춤
                            sleep = sleep[:, :in_channel_sleep, :]

                    emb[0].append(
                        torch.nn.functional.normalize(model_resp(resp)).cpu()
                    )
                    emb[1].append(
                        torch.nn.functional.normalize(model_sleep(sleep)).cpu()
                    )
                    emb[2].append(
                        torch.nn.functional.normalize(model_ekg(ekg)).cpu()
                    )

                    pbar.update()

        # 리스트 안에 들어 있는 텐서들을 concat
        emb = list(map(torch.concat, emb))  # [resp_emb, sleep_emb, ekg_emb]

        dataset_prefix = os.path.splitext(dataset_file)[0]
        save_path = os.path.join(
            path_to_save, f"{dataset_prefix}_{split}_emb.pickle"
        )

        logger.info(f"Saving embeddings for {split} to: {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(emb, f)


if __name__ == "__main__":
    generate_eval_embed()
