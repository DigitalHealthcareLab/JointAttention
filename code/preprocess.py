from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from enum import Enum, auto

from src.seed import seed_everything
from src.preprocess_videos import process_videos
seed_everything(2023)

class Task(Enum):
    IJA = auto()
    RJA_LOW = auto()
    RJA_HIGH = auto()


target_column = sys.argv[1] # label sev_cars sev_ados young_cars old_cars
task_num = int(sys.argv[2]) # 1: IJA, 2: RJA_LOW, 3: RJA_HIGH
task = Task(task_num)
task_name = task.name
data_ratio_name = sys.argv[3] # 811 622
if data_ratio_name == '811' :
    fold_num = 5
elif data_ratio_name == '622' :
    fold_num = 20


ROOT_PATH    = Path('/home/data/asd_jointattention')
DF_PATH      = ROOT_PATH.joinpath("participant_information_df.csv")
DATA_PATH    = ROOT_PATH.joinpath("raw_data_bgr").joinpath(task_name.lower())
PROC_PATH    = ROOT_PATH.joinpath("PROC_DATA").joinpath(task_name.lower())
if target_column == 'label' : 
    PROJECT_PATH = Path(f'BINARY_FOLD_{data_ratio_name}_{target_column}').joinpath(task_name)
elif target_column == 'sev_cars' : 
    PROJECT_PATH = Path(f'MULTI_FOLD_{data_ratio_name}_{target_column}').joinpath(task_name)
elif target_column == 'young_cars' : 
    PROJECT_PATH = Path(f'MULTI_FOLD_{data_ratio_name}_{target_column}').joinpath(task_name)
elif target_column == 'old_cars' :
    PROJECT_PATH = Path(f'MULTI_FOLD_{data_ratio_name}_{target_column}').joinpath(task_name)

PROC_PATH.mkdir(exist_ok=True, parents=True)
PROJECT_PATH.mkdir(exist_ok=True, parents=True)

participant_information_df = pd.read_csv(DF_PATH).rename(columns = {target_column : 'severity'}).dropna(subset=['severity']).reset_index(drop=True)


if data_ratio_name == '811' :
    test_ratio  = 0.1
    valid_ratio = 0.11 # 0.9 * 0.11 = 0.099 -> 8:1:1이 됨
    
elif data_ratio_name == '622' :
    test_ratio  = 0.2
    valid_ratio = 0.25 # 0.8 * 0.25 = 0.2 -> 6:2:2가 됨
    

task_name = task.name.lower()

participant_df = participant_information_df[['id', 'severity']]
participant_df = participant_df.dropna().drop_duplicates().reset_index(drop=True)

flatten_df = participant_information_df[['id', f"{task_name}_videos", "severity"]].explode(f"{task_name}_videos")
id_video_num_dict = dict(zip(flatten_df['id'], flatten_df[f"{task_name}_videos"]))
id_seveirty_dict = dict(zip(flatten_df['id'], flatten_df['severity']))
print(id_seveirty_dict)

flatten_dfs = []
for key, value in id_video_num_dict.items() : 
    severity = id_seveirty_dict.get(key)
    target_video_names = np.array([path.stem for path in DATA_PATH.glob(f'{key}*.mp4')])
    target_video_names = sorted(target_video_names)
    flatten_dfs.extend([[key, video_name, severity] for video_name in target_video_names])

flatten_df = pd.DataFrame(flatten_dfs, columns=['id', 'file_name', 'severity'])
print(flatten_df.head())

patient_ids = participant_df['id'].values
patient_labels = participant_df['severity'].values

for fold in range(fold_num) : 
    train_idx, test_idx = train_test_split(np.arange(len(patient_ids)), test_size=test_ratio, stratify=patient_labels)
    tr_idx, val_idx = train_test_split(train_idx, test_size=valid_ratio, stratify=patient_labels[train_idx])

    train_ids, train_labels = patient_ids[tr_idx], patient_labels[tr_idx]
    valid_ids, valid_labels = patient_ids[val_idx], patient_labels[val_idx]
    test_ids, test_labels = patient_ids[test_idx], patient_labels[test_idx]

    flatten_df[f'fold_{fold}'] = flatten_df['id'].apply(lambda x : 0 if x in train_ids else (1 if x in valid_ids else 2))


DF_PROJECT_PATH = PROJECT_PATH.joinpath('participant_information_df.csv')
flatten_df.to_csv(DF_PROJECT_PATH, index=False)

file_names = flatten_df['file_name'].values
process_videos(DATA_PATH, PROC_PATH, file_names, task_name)

