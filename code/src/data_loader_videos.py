import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import numpy as np
import pandas as pd
from enum import Enum, auto
from pathlib import Path


class VideoDataset(Dataset):
    def __init__(self, DATA_PATH : Path, DF_PATH : Path, group : str, transform = None, fold_num : int = 0):
        """
        Args:
            group: str ("train", "valid", "test")
        """
        data = pd.read_csv(DF_PATH)
        self.DATA_PATH = DATA_PATH

        if group == "train":
            self.data = data.query(f'fold_{fold_num} == 0').reset_index(drop=True)
        elif group == "valid":
            self.data = data.query(f'fold_{fold_num} == 1').reset_index(drop=True)
        elif group == "test":
            self.data = data.query(f'fold_{fold_num} == 2').reset_index(drop=True)
        else:
            raise "group must be train, valid, or test"
        assert group in {"train", "valid", "test"}
        self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target_data = self.data.iloc[idx]
        X_path = Path(self.DATA_PATH, target_data["file_name"]).with_suffix('.npy')
        X = np.load(X_path)[::3]
        y = target_data["severity"]
        return X, y


def get_loader(DATA_PATH, DF_PATH, batch_size, fold_num):
    my_transform = transforms.ToTensor()

    train_dataset = VideoDataset(DATA_PATH, DF_PATH, "train", my_transform, fold_num)
    valid_dataset = VideoDataset(DATA_PATH, DF_PATH, "valid", my_transform, fold_num)
    test_dataset  = VideoDataset(DATA_PATH, DF_PATH, "test", my_transform, fold_num)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, valid_loader, test_loader