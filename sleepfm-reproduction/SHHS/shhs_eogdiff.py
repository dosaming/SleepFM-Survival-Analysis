import numpy as np
import os
import glob
import config_shhs as config
from tqdm import tqdm

def process_eog_for_all_files(base_path):
    """
    모든 .npy 파일에서 EOG(L),EOG(R) 채널 제거
    EOG 차이 채널 추가해 새로운 폴더 생성
    """
    
    try:
        eog_l_idx = config.ALL_CHANNELS.index('EOG(L)')
        eog_r_idx = config.ALL_CHANNELS.index('EOG(R)')
    except ValueError:
        print("Error: config.ALL_CHANNELS에 'EOG(L)' 또는 'EOG(R)'이 없습니다.")
        return

    
    patient_dirs = sorted(glob.glob(os.path.join(base_path, 'X', '*')))
    print(f"총 {len(patient_dirs)}개의 환자 데이터를 처리합니다.")

    for patient_dir in tqdm(patient_dirs):
        if not os.path.isdir(patient_dir):
            continue

        epoch_files = sorted(glob.glob(os.path.join(patient_dir, '*.npy')))
        
        # 새로운 EOG 차이 신호를 저장할 경로
        new_patient_dir = patient_dir.replace('X', 'X_EOG_diff')
        os.makedirs(new_patient_dir, exist_ok=True)
        
        for file_path in epoch_files:
            try:
                # .npy 파일 로드
                data = np.load(file_path)

                # EOG 차이 신호 계산
                eog_diff_signal = data[eog_l_idx, :] - data[eog_r_idx, :]

                # 기존 EOG(L)과 EOG(R) 채널을 제외한 새로운 데이터 배열 생성
                # 인덱스 순서를 고려하여 두 채널을 제거하고 차이 신호를 추가
                
                # 제거할 인덱스 리스트
                indices_to_remove = sorted([eog_l_idx, eog_r_idx], reverse=True)
                
                # 새로운 데이터 배열
                new_data = np.delete(data, indices_to_remove, axis=0)

                # 새로운 EOG 차이 신호 추가
                new_data = np.vstack([new_data, eog_diff_signal])
                
                # 새로운 파일 이름 및 경로 설정
                new_file_path = file_path.replace('X', 'X_EOG_diff')
                
                # 새로운 파일 저장
                np.save(new_file_path, new_data)

            except Exception as e:
                print(f"파일 처리 중 오류 발생: {file_path}, 오류: {e}")
                continue

if __name__ == "__main__":
    
    data_base_path = "/ssd/kdpark/sleepfm-codebase/shhs_segments_125"
    process_eog_for_all_files(data_base_path)
