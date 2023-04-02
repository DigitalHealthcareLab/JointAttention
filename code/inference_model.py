from pathlib import Path
from enum import Enum, auto
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np
import torch
import sys
from joblib import Parallel, delayed

from src.data_loader_diagnosis_videos import get_loader
from src.custom_model_videos_res18 import Resnet18Rnn

BATCH_SIZE = 1

target_column = sys.argv[1]

class Task(Enum):
    IJA = auto()
    RJA_LOW = auto()
    RJA_HIGH = auto()

def test_trained_network(model, test_loader, device):
    model.eval()
    # Initialize the prediction and label lists(tensors)
    predlist = torch.zeros(0, dtype=torch.long, device="cpu")
    lbllist = torch.zeros(0, dtype=torch.long, device="cpu")
    alphas_arrs = []
    softmax_arrs = []

    for batch_idx, (X, y) in enumerate(test_loader, 1):
        X = X.to(device)
        y = y.long().to(device)
        output, alphas_t = model(X)

        alphas_arr = alphas_t.detach().cpu().numpy()
        alphas_arrs.extend(alphas_arr)

        pred = torch.argmax(output, dim=1)
        softmax_arr = torch.softmax(output, dim=1).detach().cpu().numpy()
        softmax_arrs.extend(softmax_arr)

        # Append batch prediction results
        predlist = torch.cat([predlist, pred.view(-1).cpu()])
        lbllist = torch.cat([lbllist, y.view(-1).cpu()])
    
    return np.array(alphas_arrs), torch.LongTensor(lbllist).numpy(), predlist.numpy(), lbllist.numpy(), np.array(softmax_arrs)


def inference_single_model(model, target_column, task_num, data_ratio_name, target_fold_num) :
    task = Task(task_num)
    task_name = task.name
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
    CHECKPOINT_PATHS = list(PROJECT_PATH.glob('*.pt'))
    CHECKPOINT_PATHS = sorted(CHECKPOINT_PATHS, key = lambda x : int(x.stem.split('_')[-1]))
    CHECKPOINT_PATH = CHECKPOINT_PATHS[-1]
    print(CHECKPOINT_PATH)
    PROC_PATH.mkdir(exist_ok=True, parents=True)
    PROJECT_PATH.mkdir(exist_ok=True, parents=True)

    PATIENCE = 7
    
    N_EPOCHS = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.to(device)

    train_loader, valid_loader, test_loader = get_loader(PROC_PATH, DF_PATH, BATCH_SIZE, target_fold_num)

    alphas_arr, labels, preds, labels_arr, softmax_arr = test_trained_network(
        model,
        test_loader,
        device
    )

    np.save(PROJECT_PATH.joinpath(f"predictions.npy"), softmax_arr)
    np.save(PROJECT_PATH.joinpath(f"labels.npy"), labels_arr)

    torch.cuda.empty_cache()


def main():
    total_num_tasks = 3
    c = 1
    if target_column == 'label' :
        OUTPUT_SIZE = 2
        DROPOUT_RATIO = 0.5
    else :
        OUTPUT_SIZE = 3
        DROPOUT_RATIO = 0.1

    model_ija = Resnet18Rnn(
        batch_size=BATCH_SIZE,
        input_size=512,
        output_size=OUTPUT_SIZE,
        seq_len=100,
        num_hiddens=512,
        num_layers=2,
        dropout=DROPOUT_RATIO,
        attention_dim=100,
    )

    model_rja =  Resnet18Rnn(
        batch_size=BATCH_SIZE,
        input_size=512,
        output_size=OUTPUT_SIZE,
        seq_len=50,
        num_hiddens=512,
        num_layers=2,
        dropout=DROPOUT_RATIO,
        attention_dim=50,
    )


    for task_num in range(1, 4) :
        task = Task(task_num)
        task_name = task.name
        for data_ratio_name, num_folds in [
                                            ['811', 5], 
                                            # ['622', 20]
                                            ] : 
            print(f"Start inference for {task_name} {data_ratio_name} {target_column} {c} / {total_num_tasks}" )

            if task_num == 1 :
                Parallel(n_jobs=5)(delayed(inference_single_model)(model_ija, target_column, task_num, data_ratio_name, target_fold_num) for target_fold_num in range(num_folds))
            else :
                Parallel(n_jobs=5)(delayed(inference_single_model)(model_rja, target_column, task_num, data_ratio_name, target_fold_num) for target_fold_num in range(num_folds))
            
            c+=1
                    


if __name__ == "__main__":
    main()