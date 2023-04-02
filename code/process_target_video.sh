#!/bin/sh

target_column=$1 # label sev_cars sev_ados young_cars old_cars

for task_num in 1 2 3
do
    # python preprocess.py $target_column $task_num 811
    bash total_process.sh $target_column $task_num 811 5
done