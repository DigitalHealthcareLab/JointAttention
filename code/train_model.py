from pathlib import Path
from enum import Enum, auto
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

from src.earlystopping import EarlyStopping
from src.data_loader_diagnosis_videos import get_loader
from src.custom_model_videos_res18 import Resnet18Rnn

class Task(Enum):
    IJA = auto()
    RJA_LOW = auto()
    RJA_HIGH = auto()

target_column = sys.argv[1] # label sev_cars sev_ados young_cars old_cars
task_num = int(sys.argv[2]) # 1: IJA, 2: RJA_LOW, 3: RJA_HIGH
task = Task(task_num)
task_name = task.name
data_ratio_name = sys.argv[3] # 811 622
target_fold_num = int(sys.argv[4]) # Under fold_num


# 경로 및 Argument 설정
ROOT_PATH    = Path('/home/data/asd_jointattention')
DATA_PATH    = ROOT_PATH.joinpath("raw_data_bgr").joinpath(task_name.lower())
PROC_PATH    = ROOT_PATH.joinpath("PROC_DATA").joinpath(task_name.lower())
if target_column == 'label' : 
    PROJECT_PATH = Path(f'BINARY_FOLD_{data_ratio_name}_{target_column}').joinpath(task_name).joinpath(f'fold_{target_fold_num}')
    OUTPUT_SIZE = 2
    DROPOUT_RATIO = 0.5
else :
    PROJECT_PATH = Path(f'MULTI_FOLD_{data_ratio_name}_{target_column}').joinpath(task_name).joinpath(f'fold_{target_fold_num}')
    OUTPUT_SIZE = 3
    DROPOUT_RATIO = 0.1
DF_PATH      = PROJECT_PATH.parent.joinpath("participant_information_df.csv")
MODEL_STEM   = f"res18rnn_{task_name.lower()}_{data_ratio_name}_weight"
PROC_PATH.mkdir(exist_ok=True, parents=True)
PROJECT_PATH.mkdir(exist_ok=True, parents=True)

PATIENCE = 7
BATCH_SIZE = 6
N_EPOCHS = 10

if task_num == 1 : 
    SEQ_LEN = 100
else:
    SEQ_LEN = 50

# Change logger save @ earlystopping.py

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s|%(levelname)s|%(message)s")

file_handler = logging.FileHandler(
    Path(PROJECT_PATH, "train_log_ija.log")
)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def train_model(
    model,
    n_epochs,
    train_loader,
    valid_loader,
    optimizer,
    criterion,
    early_stopping,
    device,
    scheduler,
):
    # initialize lists to monitor train, valid loss and accuracy
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    train_acc = []
    valid_acc = []
    logger.info(f"Training Start.")

    for epoch in range(1, n_epochs + 1):
        # train the model
        model.train()  # prep model for training
        for batch_idx, (X, y) in enumerate(train_loader, 1):
            X = X.float().to(device)
            # longtensor
            y = y.long().to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass
            output, _ = model(X)
            loss = criterion(output, y)
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

            pred = torch.argmax(output, dim=1)
            correct = (pred == y).sum().float()
            batch_acc = torch.round((correct / len(y) * 100).cpu())
            train_acc.append(batch_acc)
            logger.debug(
                f"epoch: {epoch}; batch:{batch_idx}/{len(train_loader)}; train_loss:{loss.item():.2f}"
            )
        scheduler.step()

        # validate the model#
        model.eval()  # prep model for evaluation
        for batch_idx, (X, y) in enumerate(valid_loader, 1):
            X = X.float().to(device)
            y = y.long().to(device)
            # forward pass
            output, _ = model(X)
            # calculate the loss
            loss = criterion(output, y)

            pred = torch.argmax(output, dim=1)
            correct = (pred == y).sum().float()
            batch_acc = torch.round((correct / len(y) * 100).cpu())
            valid_acc.append(batch_acc)
            # record validation loss
            valid_losses.append(loss.item())
            logger.debug(
                f"epoch: {epoch}; batch:{batch_idx}/{len(valid_loader)}; valid_loss:{loss.item():.2f}"
            )

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        epoch_train_acc = np.average(train_acc)
        epoch_valid_acc = np.average(valid_acc)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (
            f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
            + f"train_loss: {train_loss:.5f} "
            + f"train_accuracy: {epoch_train_acc:.5f} "
            + f"valid_loss: {valid_loss:.5f} "
            + f"valid_accuracy: {epoch_valid_acc:.5f} "
        )

        logger.info(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs validation loss to check if it has decreased, and if it has, it will make a checkpoint of current model
        early_stopping(valid_loss, model, epoch)

        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    return model, avg_train_losses, avg_valid_losses


def main():
    logger.info(f"{task}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path = PROJECT_PATH, model_stem = MODEL_STEM)

    train_loader, valid_loader, _ = get_loader(PROC_PATH, DF_PATH, BATCH_SIZE, target_fold_num)

    model, train_loss, valid_loss = train_model(
        model,
        N_EPOCHS,
        train_loader,
        valid_loader,
        optimizer,
        criterion,
        early_stopping,
        device,
        scheduler,
    )

if __name__ == "__main__":
    main()