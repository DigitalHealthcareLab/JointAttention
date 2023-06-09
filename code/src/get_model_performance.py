import sys
import os
root_path = os.path.abspath('..')
sys.path.append(root_path)

from pathlib import Path
from enum import Enum, auto
import pandas as pd
import numpy as np
import torch
import sys
from joblib import Parallel, delayed
import cv2
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve, 
)
from torchmetrics.classification import ConfusionMatrix


import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from src.seed import seed_everything


class Task(Enum):
    IJA = auto()
    RJA_LOW = auto()
    RJA_HIGH = auto()


class score_loader : 
    def __init__(self, result_path : Path) : 
        self.result_path = result_path
        self.total_labels = []
        self.total_preds = []
        
    def load_fold_result(self, fold_num : int) : 
        fold_path = self.result_path.joinpath(f"fold_{fold_num}")
        labels = np.load(fold_path.joinpath("labels.npy"))
        preds = np.load(fold_path.joinpath("predictions.npy"))
        self.total_labels.extend(labels)
        self.total_preds.extend(preds)
        
    def load_all_result(self, num_folds) : 
        for fold_num in range(num_folds): 
            self.load_fold_result(fold_num)


def bootstraping_auc(y_true, y_pred, n_bootstraps=1000, seed=2023):
    # yonden_cut = find_youden_index(y_true, y_pred)
    np.random.seed(seed)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices], multi_class='ovr')
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # 95% CI
    avg_score = np.mean(sorted_scores)
    ci_low = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_high = sorted_scores[int(0.975 * len(sorted_scores))]

    return avg_score, ci_low, ci_high


def bootstraping_acc(y_true, y_pred, n_bootstraps=1000, seed=2023):
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = y_pred >= 0.5
    np.random.seed(seed)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = accuracy_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # 95% CI
    avg_score = np.mean(sorted_scores)
    ci_low = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_high = sorted_scores[int(0.975 * len(sorted_scores))]

    return avg_score, ci_low, ci_high


def bootstraping_f1(y_true, y_pred, n_bootstraps=1000, seed=2023):
    y_pred = np.argmax(y_pred, axis=1)
    # y_pred = y_pred >= 0.5
    np.random.seed(seed)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = f1_score(y_true[indices], y_pred[indices], average='weighted')
        # score = f1_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # 95% CI
    avg_score = np.mean(sorted_scores)
    ci_low = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_high = sorted_scores[int(0.975 * len(sorted_scores))]

    return avg_score, ci_low, ci_high


def return_binary_scores(y_true, y_pred):
    soft_pred = y_pred[:,1]
    hard_pred = np.argmax(y_pred, axis=1)
    auroc = roc_auc_score(y_true, soft_pred)
    acc = accuracy_score(y_true, hard_pred)
    precision = precision_score(y_true, hard_pred)
    recall = recall_score(y_true, hard_pred)
    # f1 = f1_score(y_true, hard_pred)
    return auroc, acc, precision, recall #, f1


def return_multiclass_scores(y_true, y_pred):
    soft_pred = y_pred.copy()
    hard_pred = np.argmax(y_pred, axis=1)
    
    auroc = roc_auc_score(y_true, soft_pred, multi_class='ovr')
    acc = accuracy_score(y_true, hard_pred)
    precision = precision_score(y_true, hard_pred, average='weighted')
    recall = recall_score(y_true, hard_pred, average='weighted')
    # f1 = f1_score(y_true, hard_pred, average='weighted')
    return auroc, acc, precision, recall #, f1


def bootstrap_score(y_true, y_pred, n_bootstraps=3000, score_type="auc", seed=2023):
    is_binary = y_pred.shape[1] == 2   

    np.random.seed(seed)
    aurocs = []
    accs = []
    precisions = []
    recalls = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        if is_binary : 
            auroc, acc, precision, recall = return_binary_scores(y_true[indices], y_pred[indices])
        else :
            auroc, acc, precision, recall = return_multiclass_scores(y_true[indices], y_pred[indices])

        aurocs.append(auroc)
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)

    auroc_scores = sorted(np.array(aurocs))
    acc_scores = sorted(np.array(accs))
    precision_scores = sorted(np.array(precisions))
    recall_scores = sorted(np.array(recalls))


    def get_ci(scores):
        avg_score = np.mean(scores)
        ci_low = scores[int(0.025 * len(scores))]
        ci_high = scores[int(0.975 * len(scores))]
        return avg_score, ci_low, ci_high
    
    auc_avg, auc_low, auc_high = get_ci(auroc_scores)
    acc_avg, acc_low, acc_high = get_ci(acc_scores)
    precision_avg, precision_low, precision_high = get_ci(precision_scores)
    recall_avg, recall_low, recall_high = get_ci(recall_scores)
    return auc_avg, auc_low, auc_high, acc_avg, acc_low, acc_high, precision_avg, precision_low, precision_high, recall_avg, recall_low, recall_high







































"""
self.mean_tpr = np.mean(self.total_tprs, axis=0)
self.mean_tpr[-1] = 1.0
mean_auc = auc(np.linspace(0, 1, 100), self.mean_tpr)
std_auc = np.std(np.array(self.total_scores).reshape(num_folds, -1), axis=0)
std_tpr = np.std(self.total_tprs, axis=0)
self.tprs_upper = np.minimum(self.mean_tpr + std_tpr, 1)
self.tprs_lower = np.maximum(self.mean_tpr - std_tpr, 0)


def set_target_pred_label(label_idx : int) : 
    self.target_labels = np.where(np.array(self.total_labels) == label_idx, 1, 0)
    self.target_preds = np.where(np.array(self.total_preds) == label_idx, 1, 0)

def load_total_result(self, score_type = "auc") : 
    self.mean_auc = np.mean(self.total_auc)
    self.avg_score, self.ci_low, self.ci_high = self.bootstrap_score(np.array(self.total_labels), np.array(self.total_preds), score_type = score_type)
    self.mean_auc = roc_auc_score(np.array(self.total_labels), np.array(self.total_preds))
    print(f"{self.score_type if self.score_type != 'auc' else 'AUC'} {self.avg_score :.3f} ({self.ci_low:.3f}-{self.ci_high:.3f})", end = '\n ')






def draw_roc_plot(score_loader, name) :
    fig, ax = plt.subplots(figsize=(10, 10))
    score_loader.load_total_result()
    labels = np.array(score_loader.total_labels).reshape(-1)
    preds = np.array(score_loader.total_preds).reshape(-1)
    auc_score = roc_auc_score(labels, preds)
    fpr, tpr, _ = roc_curve(labels, preds, drop_intermediate=False)
    ci_min = score_loader.ci_low
    ci_max = score_loader.ci_high
    tprs_lower, tprs_upper = score_loader.tprs_lower, score_loader.tprs_upper

    ax.plot(fpr, tpr, label=f'AUROC {auc_score:.3f} (95% CI {ci_min:.3f}-{ci_max:.3f})', lw=2, alpha = 0.8)
    ax.fill_between(np.linspace(0, 1, 100), tprs_lower, tprs_upper, alpha = 0.2)
    ax.plot([0, 1], [0,1], '--', color = 'navy', lw = 2, label = None)

    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('1 - Specificity')
    ax.set_title(f'ROC Curve')
    ax.legend(loc="lower right")
    
    image_save_path = PLOT_PATH.joinpath(name)
    plt.ylim(0, 1.01)
    plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True, format='pdf')
    plt.close()
    plt.clf()


def get_score(y_true, y_pred) : 
    mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    self.total_tprs.append(interp_tpr)
    auc_score = roc_auc_score(y_true, y_pred)
    df_test = pd.DataFrame({'fpr':fpr, 'tpr':tpr})
    df_test['youden_j'] = df_test['tpr'] - df_test['fpr']
    df_test['youden_j'] = df_test['youden_j'].abs()
    return auc_score, df_test




target_column = 'label'
target_name = 'ASD_vs_TD'
data_ratio_name = '622'
task_num = 3

# data_ratio_name = '622'
if data_ratio_name == '811' :
    num_folds = 5
elif data_ratio_name == '622' : 
    num_folds = 20

task = Task(task_num)
task_name = task.name
ROOT_PATH = Path('/home/data/asd_jointattention')
DATA_PATH = ROOT_PATH.joinpath("raw_data_bgr").joinpath(task_name.lower())
PROC_PATH = ROOT_PATH.joinpath("PROC_DATA").joinpath(task_name.lower())
PLOT_PATH = Path('plots').joinpath(f"{data_ratio_name}_{target_name}")
PLOT_PATH.mkdir(exist_ok=True, parents=True)
if target_column == 'label' : 
    PROJECT_PATH = Path(f'BINARY_FOLD_{data_ratio_name}_{target_column}').joinpath(task_name)
else :
    PROJECT_PATH = Path(f'MULTI_FOLD_{data_ratio_name}_{target_column}').joinpath(task_name)

DF_PATH = PROJECT_PATH.parent.joinpath("participant_information_df.csv")
PROC_PATH.mkdir(exist_ok=True, parents=True)
PROJECT_PATH.mkdir(exist_ok=True, parents=True)

ensemble_loader = ensemble_score_loader(PROJECT_PATH)
ensemble_loader.load_all_result(num_folds)
draw_roc_plot(ensemble_loader, f"{task_name.upper()}_ROC_PLOT.pdf")
# draw_performance_plot(ensemble_loader, f"{task_name.upper()}_PERFORMANCE_PLOT.pdf")
"""