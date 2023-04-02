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

import sys
import os
root_path = os.path.abspath('..')
sys.path.append(root_path)

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss,log_loss, roc_curve, auc
import torch
from torchmetrics.classification import ConfusionMatrix
import seaborn as sns

from src.seed import seed_everything


class Task(Enum):
    IJA = auto()
    RJA_LOW = auto()
    RJA_HIGH = auto()



def func_nll(y_true, y_pred):
    nll = round(log_loss(y_true, y_pred, eps=1e-15, normalize=True),2)
    return nll

def find_youden_index(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

def func_specificity(y_true, y_pred):
    cm_generator = ConfusionMatrix(task = 'binary', num_classes=2)
    confusionmatrix = cm_generator(preds = torch.tensor(y_pred), target = torch.tensor(y_true))
    specificity = round(int(confusionmatrix[0,0])/(int(confusionmatrix[0,0])+int(confusionmatrix[0,1])), 2)
    return specificity

class ensemble_score_loader : 
    def __init__(self, result_path : Path) : 
        self.result_path = result_path
        self.total_labels = []
        self.total_preds = []
        self.total_scores = []
        self.total_dfs = []
        self.total_tprs = []

    # 95% CI with bootstrap
    def bootstrap_score(self,y_true, y_pred, n_bootstraps=1000, score_type="auc", seed=2023):
        self.score_type = score_type
        yonden_cut = find_youden_index(y_true, y_pred)
        if score_type == "auc" :
            score_func = roc_auc_score
        elif score_type == 'sensitivity':
            score_func = recall_score
            y_pred = y_pred >= 0.5
        elif score_type == 'specificity': 
            score_func = func_specificity
            y_pred = y_pred >= 0.5
        elif score_type == "accuracy" : 
            score_func = accuracy_score
            y_pred = y_pred >= 0.5
        elif score_type == "nll" : 
            score_func = func_nll
            y_pred = y_pred >= 0.5
            nll = []
        elif score_type == "brier" : 
            score_func = brier_score_loss
            y_pred = y_pred >= 0.5 

        np.random.seed(seed)
        bootstrapped_scores = []
        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = np.random.randint(0, len(y_pred), len(y_pred))
            if len(np.unique(y_true[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue
            score = score_func(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)

        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()

        # 95% CI
        avg_score = np.mean(sorted_scores)
        ci_low = sorted_scores[int(0.025 * len(sorted_scores))]
        ci_high = sorted_scores[int(0.975 * len(sorted_scores))]

        return avg_score, ci_low, ci_high
        
    def load_fold_result(self, fold_num : int) : 
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

        fold_path = self.result_path.joinpath(f"fold_{fold_num}")
        labels = np.load(fold_path.joinpath("labels.npy"))
        preds = np.load(fold_path.joinpath("predictions.npy"))[:,1]
        auc_score, df_test = get_score(labels, preds)

        self.total_labels.extend(labels)
        self.total_preds.extend(preds)
        self.total_scores.append(auc_score)
        self.total_dfs.append(df_test)

    def load_all_result(self, num_folds) : 
        for fold_num in range(num_folds): 
            self.load_fold_result(fold_num)

        self.mean_tpr = np.mean(self.total_tprs, axis=0)
        self.mean_tpr[-1] = 1.0
        mean_auc = auc(np.linspace(0, 1, 100), self.mean_tpr)
        std_auc = np.std(np.array(self.total_scores).reshape(5, -1), axis=0)
        std_tpr = np.std(self.total_tprs, axis=0)
        self.tprs_upper = np.minimum(self.mean_tpr + std_tpr, 1)
        self.tprs_lower = np.maximum(self.mean_tpr - std_tpr, 0)
        

    def load_total_result(self, score_type = "auc") : 
        self.mean_score = np.mean(self.total_scores)
        avg_score, self.ci_low, self.ci_high = self.bootstrap_score(np.array(self.total_labels), np.array(self.total_preds), score_type = score_type)
        self.avg_score = avg_score
        print(f"{self.score_type if self.score_type != 'auc' else 'AUC'} {avg_score :.3f} ({self.ci_low:.3f}-{self.ci_high:.3f})", end = '\n ')

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



def draw_performance_plot(ensemble_loader, name) : 
    score_dfs = []
    for score_type, score_name in [
                                    ['auc', 'AUC'], 
                                    ['sensitivity', 'Sensitivity'], 
                                    ['specificity', 'Specificity'], 
                                    ['accuracy', 'Accuracy'], 
                                    ] :
        ensemble_loader.load_total_result(score_type)

        labels = np.array(ensemble_loader.total_labels).reshape(-1)
        preds = np.array(ensemble_loader.total_preds).reshape(-1)
        avg_score = ensemble_loader.avg_score
        ci_low = ensemble_loader.ci_low
        ci_high = ensemble_loader.ci_high
        score_dfs.append([score_name, avg_score, ci_low, ci_high])
    score_df = pd.DataFrame(score_dfs)
    score_df.columns = ['Metric', 'Average_Score', 'CI_Low', 'CI_High']
    sns.barplot(x = 'Metric', y = 'Average_Score', data = score_df, palette='pastel')
    plt.xlabel('')
    plt.ylabel('')
    plt.ylim(0, 1.01)

    for ci_low, ci_high, x, y in zip(score_df['CI_Low'], score_df['CI_High'], score_df.index, score_df['Average_Score']) : 
        plt.plot([x, x], [ci_low, ci_high], color = 'k', linewidth = 1)
        plt.plot(x, y, 'o', color = 'k')
    
    image_save_path = PLOT_PATH.joinpath(name)
    plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True, format='pdf')
    plt.close()
    plt.clf()


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
draw_performance_plot(ensemble_loader, f"{task_name.upper()}_PERFORMANCE_PLOT.pdf")
