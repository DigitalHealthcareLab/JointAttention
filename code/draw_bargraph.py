from pathlib import Path
import numpy as np
import pandas as pd
from enum import Enum, auto
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    auc,
    roc_curve,
)

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.get_model_performance import score_loader
from src.get_model_performance import bootstrap_score


def load_results(target_column, task_num, num_folds) :
    class Task(Enum):
        IJA = auto()
        RJA_LOW = auto()
        RJA_HIGH = auto()

    task = Task(task_num)
    task_name = task.name

    ROOT_PATH    = Path('/mnt/2021_NIA_data/jointattention')
    if target_column == 'label' : 
        PROJECT_PATH = Path(f'BINARY_FOLD_{target_column}').joinpath(task_name)
    else :
        PROJECT_PATH = Path(f'MULTI_FOLD_{target_column}').joinpath(task_name)

    loader = score_loader(PROJECT_PATH)
    loader.load_all_result(num_folds)
    labels = np.array(loader.total_labels)
    preds = np.array(loader.total_preds)
    num_labels = len(np.unique(labels))
    return PROJECT_PATH, labels, preds, num_labels


def draw_performance_plot(score_df, image_save_path : Path, title) : 
    # plt.figure(figsize = (12,12))
    score_df2 = score_df.melt(id_vars=['task'], var_name='metric', value_name='score')
    score_df2 = score_df2[score_df2['metric'].apply(lambda x : 'CI' not in x)]
    score_df2['score'] = score_df2['score'] * 100
    bar = sns.barplot(x = 'metric', y='score', hue='task', data=score_df2, palette='pastel')
    plt.title(title)
    target_score_columns = ['AUROC', 'Accuracy', 'Precision', 'Recall']
    bar.set_xticklabels(target_score_columns)

    for task_idx, [task_name, new_task_name] in enumerate([['IJA', 'IJA'],
                                                            ['RJA_LOW','RJA low'] ,
                                                            ['RJA_HIGH', 'RJA high']], -1) :
        for score_idx, score_metric in enumerate(target_score_columns) : 
            score = score_df.query('task == @new_task_name')[score_metric].values[0] * 100
            ci_low = score_df.query('task == @new_task_name')[f'{score_metric}_CI_low'].values[0] * 100
            ci_high = score_df.query('task == @new_task_name')[f'{score_metric}_CI_high'].values[0] * 100
            xposition = score_idx + task_idx * 0.25
            print(new_task_name, score_metric, f"{score:.1f}% (95% CI, {ci_low:.1f}-{ci_high:.1f})")
            bar.text(xposition, ci_high + 2, f'{score:.1f}', ha='center', va = 'center',  color='black', fontsize=6)
            # bar.text(xposition, ci_high + 3, f'{ci_high:.1f}', ha='center', va='top', color='black', fontsize=8)
            # bar.text(xposition, ci_low - 3, f'{ci_low:.1f}', ha='center', va='bottom', color='black', fontsize=8)
            plt.plot([xposition, xposition], [ci_low , ci_high], color="black")

    plt.xlabel('') 
    plt.ylabel('AUROC/Accuracy/Precision/Recall(%)', fontsize=12)
    plt.ylim(0, 105)
    plt.legend(bbox_to_anchor=(0.9, 1.15), loc=2, borderaxespad=0., fontsize=8, frameon=False)
    image_save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(image_save_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True, format='pdf')
    plt.close()


def return_score_results(labels, preds, auc_scores) : 
    # hard_preds = np.argmax(preds, axis=1)
    auc, auc_ci_low, auc_ci_high,\
    accuracy, acc_ci_low, acc_ci_high,\
    precision, precision_ci_low, precision_ci_high,\
    recall, recall_ci_low, recall_ci_high  = bootstrap_score(labels, preds, 5000)
    return [auc, auc_ci_low, auc_ci_high, accuracy, acc_ci_low, acc_ci_high, precision, precision_ci_low, precision_ci_high, recall, recall_ci_low, recall_ci_high]


def load_task_result(target_column, task_num, num_folds) :
    PROJECT_PATH, labels, preds, num_labels = load_results(target_column, task_num, num_folds)
    score_results = return_score_results(labels, preds, auc)
    return PROJECT_PATH, score_results


if __name__ == '__main__' :
    columns = ['task', 'AUROC', 'AUROC_CI_low', 'AUROC_CI_high', 'Accuracy', 'Accuracy_CI_low', 'Accuracy_CI_high', 'Precision', 'Precision_CI_low', 'Precision_CI_high', 'Recall', 'Recall_CI_low', 'Recall_CI_high']
    for target_column, num_folds in [
                            # ['label', 20],
                            # ['sev_cars', 20],
                            ["sev_ados", 10],

                            # 'young_cars', 
                            # 'old_cars',
                            # ['young_ados',10],
                            # ['old_ados',10],

                            ] :
        print()
        print(target_column)
        if target_column == 'label' : 
            total_results = []
            score_results = []
            for task_num, [task_name, new_task_name] in enumerate([
                                        ['IJA', 'IJA'],
                                        ['RJA_LOW','RJA low'] ,
                                        ['RJA_HIGH', 'RJA high']
                                        ],   1) :

                PROJECT_PATH, score_result = load_task_result(target_column, task_num, num_folds)
            
                score_results.append([new_task_name] + score_result)
            title = 'ASD vs TD'
            image_save_path = Path('plots').joinpath(PROJECT_PATH.parent.name).joinpath(f"BINARY_CLF_SCORE.pdf")

            score_df = pd.DataFrame(score_results, columns = columns)
            draw_performance_plot(score_df, image_save_path, title)
        else : 
            score_dfs = []
            for task_num, [task_name, new_task_name] in enumerate([
                                        ['IJA', 'IJA'],
                                        ['RJA_LOW','RJA low'] ,
                                        ['RJA_HIGH', 'RJA high']], 1) :
                PROJECT_PATH, score_results = load_task_result(target_column, task_num, num_folds)
                title = 'non-ASD vs mild-moderate ASD vs severe ASD'
                score_df = pd.DataFrame([[new_task_name] + score_results], columns = columns)
                score_dfs.append(score_df)
            score_df = pd.concat(score_dfs)
            image_save_path = Path('plots').joinpath(PROJECT_PATH.parent.name).joinpath(f"{target_column}_SCORE3.pdf")                
            image_save_path.parent.mkdir(parents=True, exist_ok=True)
            draw_performance_plot(score_df, image_save_path, title)
            

            