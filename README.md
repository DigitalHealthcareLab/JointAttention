## Project name: JointAttention
* author: Chanyoung ko
* date: 2022-03-31
* python version: 3.6.8
* opencv version: 2.4.5

## Objective
- Create a classification model for ASD(Autism Spectrum Disorder) vs TD(Typical development)
- Create a classification model for non-ASD vs mild-moderate ASD vs severe ASD

## Project structure
* checkpoint
* code
* data
** proc_data
    1. proc_ija
    2. proc_rja_high
    3. proc_rja_low
** raw_data
    1. ija
    2. rja_high
    3. rja_low 
** raw_data_png
    1. ija
    2. rja_high
    3. rja_low

## Dataset
### [`data/dataset/ija_videofile_with_dx.csv`](data/dataset/ija_videofile_with_dx.csv)
### [`data/dataset/ija_videofile_with_dx.csv`](data/dataset/ija_videofile_with_sev.csv)
### [`data/dataset/rja_high_videofile_with_dx.csv`](data/dataset/rja_high_videofile_with_dx.csv)
### [`data/dataset/rja_high_videofile_with_dx.csv`](data/dataset/rja_high_videofile_with_sev.csv)
### [`data/dataset/rja_low_videofile_with_dx.csv`](data/dataset/rja_low_videofile_with_dx.csv)
### [`data/dataset/rja_low_videofile_with_dx.csv`](data/dataset/rja_low_videofile_with_sev.csv)
         
* Labels
    Binary labels: 0 - TD, 1 - ASD
    Multi-labels: 0 - non-ASD, 1 - mild-moderate ASD, 2 - severe ASD
    Cut-offs for multi-labels are based on on ADOS CSS or CARS cut-off scores# JointAttention
