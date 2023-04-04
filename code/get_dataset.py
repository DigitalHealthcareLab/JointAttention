from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
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
fold_num = 10
valid_ratio = 0.11


# 경로 및 Argument 설정
ROOT_PATH    = Path('/mnt/2021_NIA_data/jointattention')
DF_PATH      = ROOT_PATH.joinpath("participant_information_df.csv")
DATA_PATH    = ROOT_PATH.joinpath("raw_data_bgr").joinpath(task_name.lower())
PROC_PATH    = ROOT_PATH.joinpath("PROC_DATA").joinpath(task_name.lower())
if target_column   == 'label' : 
    PROJECT_PATH = Path(f'BINARY_FOLD_{target_column}').joinpath(task_name)
else : 
    PROJECT_PATH = Path(f'MULTI_FOLD_{target_column}').joinpath(task_name)
PROC_PATH.mkdir(exist_ok=True, parents=True)
PROJECT_PATH.mkdir(exist_ok=True, parents=True)

participant_information_df = pd.read_csv(DF_PATH).rename(columns = {target_column : 'severity'}).dropna(subset=['severity']).reset_index(drop=True)    

task_name = task.name.lower()

### 여기서부터는 자동으로 돌아감
participant_df = participant_information_df[['id', 'severity']]
participant_df = participant_df.dropna().drop_duplicates().reset_index(drop=True)

flatten_df = participant_information_df[['id', f"{task_name}_videos", "severity"]].explode(f"{task_name}_videos")
id_video_num_dict = dict(zip(flatten_df['id'], flatten_df[f"{task_name}_videos"]))
id_seveirty_dict = dict(zip(flatten_df['id'], flatten_df['severity']))

flatten_dfs = []
for key, value in id_video_num_dict.items() : 
    severity = id_seveirty_dict.get(key)
    target_video_names = np.array([path.stem for path in DATA_PATH.glob(f'{key}*.mp4')])
    target_video_names = sorted(target_video_names)
    flatten_dfs.extend([[key, video_name, severity] for video_name in target_video_names])

flatten_df = pd.DataFrame(flatten_dfs, columns=['id', 'file_name', 'severity'])

patient_ids = participant_df['id'].values
patient_labels = participant_df['severity'].values


skf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=2023)
for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, patient_labels)) :
    tr_idx, val_idx = train_test_split(train_idx, test_size=valid_ratio, stratify=patient_labels[train_idx])

    train_ids, train_labels = patient_ids[tr_idx], patient_labels[tr_idx]
    valid_ids, valid_labels = patient_ids[val_idx], patient_labels[val_idx]
    test_ids, test_labels = patient_ids[test_idx], patient_labels[test_idx]

    flatten_df[f'fold_{fold}'] = flatten_df['id'].apply(lambda x : 0 if x in train_ids else (1 if x in valid_ids else 2))


DF_PROJECT_PATH = PROJECT_PATH.joinpath('participant_information_df.csv')
flatten_df.to_csv(DF_PROJECT_PATH, index=False)

file_names = flatten_df['file_name'].values
process_videos(DATA_PATH, PROC_PATH, file_names, task_name)




#### 아래는 데이터 전처리를 위해서 participant_info_df를 수정하는 과정임
# from pathlib import Path
# import pandas as pd
# import numpy as np
# ROOT_PATH    = Path('/home/data/asd_jointattention')
# DF_PATH      = ROOT_PATH.joinpath("participant_information_df.csv")
# df = pd.read_csv(DF_PATH).drop(columns = ['young_group'])
# df['young_cars'] = df.apply(lambda x : x['sev_cars'] if x['age'] <= 48 else np.nan, axis=1)
# df['old_cars'] = df.apply(lambda x : x['sev_cars'] if x['age'] > 48 else np.nan, axis=1)
# df.to_csv(DF_PATH, index=False)


