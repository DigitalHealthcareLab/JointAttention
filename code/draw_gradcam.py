import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import copy
import cv2
import matplotlib.pylab as plt
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum, auto
from torch.utils.data import DataLoader
from joblib import Parallel, delayed

from src.custom_model_videos_res18 import Resnet18Rnn
from src.custom_model_videos_gradcam import GradCamModel
from src.data_loader_videos import VideoDataset


class Task(Enum):
    IJA = auto()
    RJA_LOW = auto()
    RJA_HIGH = auto()

def set_paths(target_name, task_num) : 
    task = Task(task_num)
    task_name = task.name

    ROOT_PATH    = Path('/home/data/asd_jointattention')
    DATA_PATH    = ROOT_PATH.joinpath("raw_data_bgr").joinpath(task_name.lower())
    PROC_PATH    = ROOT_PATH.joinpath("PROC_DATA").joinpath(task_name.lower())
    TARGET_ROOT_PATH = Path(f'BINARY_FOLD_{target_name}') if target_name == 'label' else Path(f'MULTI_FOLD_{target_name}')
    PROJECT_PATH = TARGET_ROOT_PATH.joinpath(task_name)
    DF_PATH      = PROJECT_PATH.joinpath("participant_information_df.csv")
    return task, DATA_PATH, PROC_PATH, PROJECT_PATH, DF_PATH

def get_frames(array_path : Path) : 
    array = np.load(array_path)
    frames = array[::3]
    return frames


def process_single_image(X, y, file_idx, model_ckpt) : 
    X = X.float().to(device)
    y = y.long().to(device)
    patient_id = patient_ids[file_idx]
    file_name = file_names[file_idx]

    model = Resnet18Rnn(
                    batch_size=BATCH_SIZE,
                    input_size=512,
                    output_size=OUTPUT_SIZE,
                    seq_len=SEQ_LEN,
                    num_hiddens=512,
                    num_layers=2,
                    dropout=DROPOUT_RATIO,
                    attention_dim=SEQ_LEN,
                )

    model.load_state_dict(torch.load(model_ckpt))
    for param in model.parameters():
        param.requires_grad = True
    model.to(device)

    gradcam_model = GradCamModel(model)
    gradcam_model.to(device)
    gradcam_model.train()



    # print(f'patient_id : {patient_id}, file_name : {file_name}')

    file_save_path = save_path.joinpath(target_name, task_name, f"fold_{fold_num}", str(patient_id), file_name).with_suffix(".npy")
    file_save_path.parent.mkdir(parents=True, exist_ok=True)
    

    output = gradcam_model(X)
    c = y

    gradcam_model.zero_grad()
    output[:, c].backward()
    gradients = gradcam_model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 3, 4])
    activations = gradcam_model.get_activations(X).detach()
    channels, n_frames = activations.shape[1], activations.shape[2]
    for i in range(channels):
        for j in range(n_frames):
            activations[:,i,j,:,:]*= pooled_gradients[i,j]
            
    heatmap = torch.mean(activations, dim=1).squeeze().cpu()
    heatmap = np.maximum(heatmap,0)
    for i in range(n_frames):
        heatmap[i]/=torch.max(heatmap[i])
    heatmap = heatmap.numpy()

    input_img = X.squeeze().cpu().numpy()
    input_img = np.transpose(input_img, (0,2,3,1))

    input_img = (input_img - np.min(input_img)) / (np.max(input_img) - np.min(input_img))
    input_img = np.uint8(255*input_img)
    # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    heatmap = np.uint8(255*heatmap)
    np.save(file_save_path, heatmap)
    np.save(file_save_path.with_name(f'{file_name}_raw.npy'), input_img)


for target_name in [
                'label', 
                'sev_ados'
                ] : 
    for task_num in range(1,4) : 
        task, DATA_PATH, PROC_PATH, PROJECT_PATH, DF_PATH = set_paths(target_name, task_num)
        task_name = task.name
        BATCH_SIZE = 1
        OUTPUT_SIZE = 2 if target_name == 'label' else 3
        SEQ_LEN = 100 if task_num == 1 else 50
        DROPOUT_RATIO = 0.5 if target_name == 'label' else 0.1
        device = 'cuda'
        fold_num = 0
        for fold_num in range(10) : 
            
            my_transform = transforms.ToTensor()
            total_dataset = VideoDataset(PROC_PATH, DF_PATH, "", my_transform, fold_num)
            total_loader = DataLoader(
                                    total_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
                                    )

            FOLD_PATH = PROJECT_PATH.joinpath(f'fold_{fold_num}')
            model_ckpts = np.array(sorted(list(FOLD_PATH.glob('*.pt')), key = lambda x : int(x.stem.split('_')[-1])))
            model_ckpt = model_ckpts[-1]

            save_path = Path('GradCAM')

            df = total_loader.dataset.data

            patient_ids = df['id'].values
            file_names = df['file_name'].values


            Parallel(n_jobs=12)(delayed(process_single_image)(X, y, file_idx, model_ckpt) for file_idx, (X, y) in tqdm(enumerate(total_loader), total=len(total_loader)))