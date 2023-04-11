## Project name: JointAttention
* author: Chanyoung Ko
* date: 2022-03-31
* python version: 3.6.8
* opencv version: 4.7.0

## Objective
- Create a classification model for ASD(Autism Spectrum Disorder) vs TD(Typical Development)
- Create a classification model for non-ASD vs mild-moderate ASD vs severe ASD

## Project structure
* code
    * src 
    * data
        * raw_data
            1. ija
            2. rja_high
            3. rja_low 
        * proc_data
            1. proc_ija
            2. proc_rja_high
            3. proc_rja_low
    

## Dataset
### [`dataset/participant_information_df.csv`](dataset/participant_information_df.csv)
      
## Labels
* Binary labels: 0 - TD, 1 - ASD
* Multi-labels: 0 - non-ASD, 1 - mild-moderate ASD, 2 - severe ASD
* Cut-offs for multi-labels are based on on ADOS CSS or CARS cut-off scores

## How to run codes
* Run 'process_multiple_targets.sh to train and validate models
* Run 'draw_bargraph.py' to get graphical performance measure results
* Run 'draw_clustermap.py' to draw cluster maps using attention weights
* Run 'draw_gradcam.py' to draw gradcam on RGB videos used in model training
