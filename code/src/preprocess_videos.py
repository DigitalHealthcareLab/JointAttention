import cv2
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

def get_video_path(DATA_PATH, video_name) -> Path:
    return DATA_PATH.joinpath(video_name).with_suffix('.mp4')

def get_npy_output_path(PROC_PATH, video_name) -> Path:
    return PROC_PATH.joinpath(video_name).with_suffix('.npy')

def read_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype("uint8"))

    fc = 0
    ret = True

    while fc < frameCount and ret:
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()

    return np.transpose(buf, (0, 3, 2, 1))


def preproc_transform(video_arr, target_length):
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(3),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return torch.cat(
        [
            preprocess(frame.transpose(2, 1, 0)).unsqueeze(0)
            for frame in pad_frame(video_arr, target_length)
        ],
        axis=0,
    )


def pad_frame(video_arr, target_length=300):
    length = video_arr.shape[0]
    if length >= target_length:
        new_video_arr = video_arr[:target_length]
    elif length < target_length:
        new_video_arr = pad_along_axis(video_arr, target_length)
    return new_video_arr 


def pad_along_axis(array, target_length, axis=0):
    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode="constant", constant_values=0)

    

def preprocess_single_video(DATA_PATH : Path, PROC_PATH : Path, video_name : str, task_name) : 
    video_path = get_video_path(DATA_PATH, video_name)
    array_save_path = get_npy_output_path(PROC_PATH, video_name)
    if array_save_path.exists():
        return
    if task_name == 'ija' : 
        target_length = 300
    else : 
        target_length = 150
    video_arr = read_video(video_path)
    preproc_arr = preproc_transform(video_arr, target_length)
    np.save(array_save_path, preproc_arr)
    print(f"Saved {array_save_path} | {preproc_arr.shape}")

def process_videos(DATA_PATH : Path, PROC_PATH : Path, video_names : np.array, task_name) : 
    Parallel(n_jobs=10)(delayed(preprocess_single_video)(DATA_PATH, PROC_PATH, video_name, task_name) for video_name in tqdm(video_names, total = len(video_names)))
