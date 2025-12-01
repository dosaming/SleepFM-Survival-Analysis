
import configparser
import os
import types

_FILENAME = None
_PARAM = {}

CONFIG = types.SimpleNamespace(
    FILENAME=_FILENAME,
    DATASETS=_PARAM.get("datasets", "datasets"),
    OUTPUT=_PARAM.get("output", "output"),
    CACHE=_PARAM.get("cache", ".cache"),
)

PATH_TO_RAW_DATA = "#_SHHS/polysomnography/edfs/shhs1"
PATH_TO_PROCESSED_DATA = "/ssd/kdpark/sleepfm-codebase/shhs_segments_125"

DATASET_PICKLE_PATH = "/ssd/kdpark/sleepfm-codebase/shhs_segments_125/dataset.pickle"
DATASET_EVENTS_PATH = "/ssd/kdpark/sleepfm-codebase/shhs_segments_125/dataset_events_-1.pickle"

EMBED_SAVE_PATH = "/ssd/kdpark/sleepfm-codebase/outputs_shhs_list/my_run/embeddings"


LABELS_DICT = {
    "Wake": 0, 
    "Stage 1": 1, 
    
    "Stage 2": 2, 
    "Stage 3": 3, 
    "REM": 4
}
INV_LABELS_DICT = {v: k for k, v in LABELS_DICT.items()}

MODALITY_TYPES = ["respiratory", "sleep_stages", "ekg"]
CLASS_LABELS = ["Wake", "Stage 1", "Stage 2", "Stage 3", "REM"]
NUM_CLASSES = 5

EVENT_TO_ID = {
    "W": 0, "Wake": 0, "Wake|0": 0,
    "N1": 1, "Stage 1": 1, "Stage 1 sleep|1": 1,
    "N2": 2, "Stage 2": 2, "Stage 2 sleep|2": 2,
    "N3": 3, "Stage 3": 3, "Stage 3 sleep|3": 3, "Stage 4": 3, "Stage 4 sleep|4": 3,
    "REM": 4, "REM sleep": 4, "REM sleep|5": 4,
}

LABEL_MAP = {
    # SleepFM 
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

    # SHHS
    "Wake|0": "Wake",
    "Stage 1 sleep|1": "Stage 1",
    "Stage 2 sleep|2": "Stage 2",
    "Stage 3 sleep|3": "Stage 3",
    "Stage 4 sleep|4": "Stage 3",  
    "REM sleep|5": "REM",
}


ALL_CHANNELS = ['SaO2', 'H.R.', 'EEG(sec)', 'ECG', 'EMG', 'EOG(L)', 'EOG(R)', 'EEG', 'THOR RES', 'ABDO RES', 'POSITION', 'LIGHT']



CHANNEL_DATA = {
    "Respiratory": ["THOR RES", "ABDO RES", "SaO2"],
    "Sleep_Stages": ["EEG", "EEG(sec)", "EOG(L)", "EOG(R)", "EMG"],
    "EKG": ["ECG"], 
    }


CHANNEL_DATA_IDS = {
    "Respiratory": [ALL_CHANNELS.index(item) for item in CHANNEL_DATA["Respiratory"]], 
    "Sleep_Stages": [ALL_CHANNELS.index(item) for item in CHANNEL_DATA["Sleep_Stages"]], 
    "EKG": [ALL_CHANNELS.index(item) for item in CHANNEL_DATA["EKG"]], 
 }
