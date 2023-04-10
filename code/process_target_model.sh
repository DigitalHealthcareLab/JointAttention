#!/bin/sh

target_column=$1 # label sev_cars sev_ados young_cars old_cars
task_num=$2 # 1: IJA, 2: RJA_LOW, 3: RJA_HIGH
data_ratio_name=$3 # 811 622
fold_num=$4 # 5

# max_fold_num=$(expr $fold_num - 1)
# for i in $(seq 0 $max_fold_num)
# do
#     echo "python train_model.py $target_column $task_num $data_ratio_name $i"
#     python train_model.py $target_column $task_num $data_ratio_name $i
# done

python inference_model.py $target_column 0
