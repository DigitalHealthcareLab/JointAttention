from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

dlib_root_path = Path('DLIB_RESULT')
dlib_data_path = dlib_root_path.joinpath('label')
image_save_path = dlib_root_path.joinpath('result')
image_save_path.mkdir(exist_ok=True)

gradcam_root_path = Path('GradCAM')
gradcam_data_path = gradcam_root_path.joinpath('label')

def point_image(video_dlib, video_gradcam, seq) : 
    cmap = plt.get_cmap('jet')
    seq_dlib = np.array(video_dlib[seq])
    seq_gradcam = video_gradcam[seq]
    seq_gradcam = cv2.resize(seq_gradcam, (224, 224))
    seq_gradcam = cmap(seq_gradcam)
    seq_gradcam = np.uint8(seq_gradcam*255)
    seq_dlibs = seq_dlib.reshape(seq_dlib.shape[0]//68, 68, 3)

    if np.array_equal(seq_dlibs, np.ones_like(seq_dlibs)) :
        return seq_gradcam
    
    for seq_dlib in seq_dlibs : 
        for point, x, y in seq_dlib : 
            x, y = int(x), int(y)
            seq_gradcam = cv2.circle(seq_gradcam, (x,y), 3, (0,0,0), -1)
    
    return seq_gradcam


def save_video(video_path, seq_len) : 
    
    video_name = video_path.stem
    video_dlib_path = video_path.joinpath(f"{video_name}_raw_DLIB.npy")
    video_gradcam_path = patient_gradcam_path.joinpath(f"{video_name}.npy")
    video_dlib = np.load(video_dlib_path, allow_pickle=True)
    video_gradcam = np.load(video_gradcam_path)
    for seq in range(seq_len) : 
        # seq_pdf_save_path = image_save_path.joinpath(task_name, patient_id, video_name, f"fold_{fold_num}", "pdf", f"{video_name}_{seq:03}.pdf")
        seq_png_save_path = image_save_path.joinpath(task_name, patient_id, video_name, f"fold_{fold_num}", "png", f"{video_name}_{seq:03}.png")

        if seq_png_save_path.exists() : 
            continue

        # seq_pdf_save_path.parent.mkdir(exist_ok=True, parents=True)
        seq_png_save_path.parent.mkdir(exist_ok=True, parents=True)
        new_gradcam = point_image(video_dlib, video_gradcam, seq)
        plt.imshow(new_gradcam)
        plt.axis('off')
        # plt.savefig(seq_pdf_save_path, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True, format='pdf')
        plt.savefig(seq_png_save_path, bbox_inches='tight', pad_inches=0, dpi=100, transparent=True, format='png')

# for task_name in ['IJA', 'RJA_LOW', 'RJA_HIGH'] : 

task_name = 'RJA_HIGH'
if task_name == 'IJA' : 
    participants =[ ['TD', 'C129'], ['ASD', 'B702']]
    inner_jobs = 10
    seq_len = 100
elif task_name == 'RJA_LOW' : 
    participants =[ ['TD', 'C129'], ['ASD', 'B704']]
    inner_jobs = 5
    seq_len = 50
elif task_name == 'RJA_HIGH' :
    participants =[ ['TD', 'C129'], ['ASD', 'D710']]
    inner_jobs = 5
    seq_len = 50
    for fold_num in tqdm(range(10)) :
        for data_class, patient_id in participants : 
            patient_dlib_path = dlib_data_path.joinpath(task_name, f"fold_{fold_num}", patient_id)
            patient_gradcam_path = gradcam_data_path.joinpath(task_name, f"fold_{fold_num}", patient_id)
            patient_video_paths = list(patient_dlib_path.glob('*'))
            patient_video_paths = sorted(patient_video_paths, key=lambda x: int(x.stem.split('_')[-1]))
            patient_video_paths = np.array([patient_video_paths]).reshape(-1)
            
            Parallel(n_jobs=inner_jobs)(delayed(save_video)(video_path, seq_len) for video_path in patient_video_paths)