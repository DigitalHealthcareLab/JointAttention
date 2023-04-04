from enum import Enum, auto
from pathlib import Path
import numpy as np
import cv2
import torch
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from data_loader_sev_frames import get_loader
from custom_model_gradcam_res18 import GradCamModel
import matplotlib.pyplot as plt
import myutils
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Task(Enum):
    IJA = auto()
    RJA_LOW = auto()
    RJA_HIGH = auto()


PROJECT_PATH = Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, "data")
FIG_PATH = Path(PROJECT_PATH, "figures")
CHECKPOINT_PATH = Path(
    PROJECT_PATH,
    "checkpoint/best_diagnosis/res18rnn_rjahigh_diagnosis_811_221123_weight_3.pt",
)  # specify which weight

BATCH_SIZE = 1
TIME_STEPS = 50  # 300/3 if IJA, 150/3 if RJA_high & RJA_low


def test_gradcam_network(model, xb, yb, device):
    model.eval()
    torch.backends.cudnn.enabled = False
    model.zero_grad()
    xb = xb.to(device)
    # yb = yb.to(device)
    output = model(xb)

    c = yb
    activations = model.get_activations(xb).detach()

    output[:, c].backward()
    gradients = model.get_activations_gradient()
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 3, 4])
    # print(gradients.shape) #[1, 512, 50, 7, 7]
    # print(pooled_gradients.shape) #[512, 50]

    activations = model.get_activations(xb).detach()
    channels, TIME_STEPS = activations.shape[1], activations.shape[2]
    for i in range(channels):
        for j in range(TIME_STEPS):
            activations[:, i, j, :, :] *= pooled_gradients[i, j]

    heatmap = torch.mean(activations, dim=1).squeeze().cpu()
    heatmap = np.maximum(heatmap, 0)
    for i in range(TIME_STEPS):
        heatmap[i] /= torch.max(heatmap[i])
    heatmap = heatmap.numpy()
    return heatmap


def draw_gradcam(imgs, heatmap, png_name):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    plt.figure(figsize=(20, 10))
    x = 0.004
    count = 0
    n_layout_cnt = 10
    n_step = int(TIME_STEPS / n_layout_cnt)
    for ii in range(0, TIME_STEPS, n_step):  # 4 frames
        count += 1
        img_ = np.array(myutils.denormalize(imgs[ii].cpu(), mean, std))
        plt.subplot(2, 5, count)
        heatmap_ = cv2.resize(heatmap[ii], (img_.shape[1], img_.shape[0]))
        heatmap_ = np.uint8(255 * (1 - heatmap_))
        heatmap_ = cv2.applyColorMap(heatmap_, cv2.COLORMAP_JET)

        superimposed_img = heatmap_ * 0.8 + img_
        superimposed_img = superimposed_img / np.max(superimposed_img)
        #     cv2.imwrite('./map.jpg', superimposed_img)
        plt.axis("off")
        plt.imshow(superimposed_img)
        plt.title(png_name)

    plt.savefig(Path(FIG_PATH / f"gradcam/{task.name.lower()}", png_name))

    plt.figure(figsize=(20, 10))
    count = 0
    for ii in range(0, TIME_STEPS, n_step):  # 20, 24
        count += 1
        plt.subplot(2, 5, count)
        plt.imshow(myutils.denormalize(imgs[ii], mean, std))
        x = plt.axis("off")

    plt.savefig(Path(FIG_PATH / f"gradcam/{task.name.lower()}", png_name + "_original"))


def main(task: Task):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # print("Current cuda device:", torch.cuda.device_count())

    params_model = {
        "num_classes": 2,
        "dr_rate": 0.1,
        "rnn_num_layers": 2,
        "rnn_hidden_size": 512,
        "timesteps": TIME_STEPS,
        "checkpoint_path": CHECKPOINT_PATH,
    }

    model = GradCamModel(params_model).to(device)
    _, _, test_loader = get_loader(task, BATCH_SIZE, DATA_PATH)

    for i in range(len(test_loader.dataset.data)):
        test_loader.dataset.data.iloc[i]
        imgs, y = test_loader.dataset.get_frame(i)
        imgs = imgs.unsqueeze(0)
        y = [y]

        heatmap = test_gradcam_network(model, imgs, y, device)

        filename = test_loader.dataset.data.iloc[i]["file_name"]
        label = test_loader.dataset.data.iloc[i]["label"]
        if label == 0:
            label = "TD"
        else:
            label = "ASD"
        png_name = f"{filename}_{label}".replace(".mp4", "")
        print(png_name)

        draw_gradcam(imgs.squeeze(0), heatmap, png_name)


#%%
if __name__ == "__main__":
    task = Task.RJA_HIGH 
    main(task)
