# s2s1osaka
WARNING: Due to before code refactoring,  preprocessing and feature engineering are time consuming about 6 hours with standard specification machine.   

We archived 13 out of 1564 sign-uped teams in The ACM RecSys Challenge 2019 with Public score 0.673804 and Final score 0.671117. This publication of code is to build the solution we came up with. Also, our team consists of `yasuo.yamamoto.jp@gmail.com` and `axel.x12x@gmail.com`.  

# Requirements
- Python 3.6.7 :: Anaconda custom (64-bit)
- numpy 1.15.4
- pandas 0.23.4
- lightgbm 2.2.1

# Dataset
First, you must place this challenge's dataset in `./data_v2/` folder.
```
./data_v2/train.csv
./data_v2/test.csv
./data_v2/item_metadata.csv
./data_v2/submission_popular.csv
```

# How to check this repository as a code walkthrough
You can execute pipeline with 1/10 training dataset as follow command to evaluate this repository.  
```
$ python run.py --cehck
```
Then, you can see a messages as follow to confirm getting pipeline done normally.
Memo: It's not problem that code walkthrough occurs a warning massages like `numpy/core/.. RuntimeWarning: ...` because of using shrinked training dataset.    
```
starting
... loading dataset
... preprocessing
... ... CityForSessionStep
... ... ItemPropsVector
... ... View2viewCounter
... ... BayesLikelihood
... ... ImpressionScore
... create rows for training as X
... set dummy variables for device
... feature engineering to X
... ... Durations
... ... JustClickout
... expanding X
... ... Record2Impression
... feature engineering to expanded X
... ... EncodingForCategories
... ... Decision Making Process
... ... ... Attention and Perceptual Encoding
... ... ... Information Acquisition and Evaluation
... ... BySession as Motivation
... ... ByItem
... ... ByLocation
... ... JustBeforeClickout
... set interactions between features
... ... Polinomials
... target variable
... ... TargetVariable
... local validation
[LightGBM] [Warning] Find whitespaces in feature_names, replace with underlines
[LightGBM] [Info] Total Bins 13440
[LightGBM] [Info] Number of data: 2070576, number of used features: 186
[LightGBM] [Warning] Find whitespaces in feature_names, replace with underlines
[1]	valid_0's ndcg@1: 0.531856	valid_0's ndcg@3: 0.600321	valid_0's ndcg@5: 0.637807
[2]	valid_0's ndcg@1: 0.54882	valid_0's ndcg@3: 0.609377	valid_0's ndcg@5: 0.645577
[3]	valid_0's ndcg@1: 0.548578	valid_0's ndcg@3: 0.610547	valid_0's ndcg@5: 0.647321
[4]	valid_0's ndcg@1: 0.549225	valid_0's ndcg@3: 0.610888	valid_0's ndcg@5: 0.647868
[5]	valid_0's ndcg@1: 0.549508	valid_0's ndcg@3: 0.610983	valid_0's ndcg@5: 0.648551
[6]	valid_0's ndcg@1: 0.549669	valid_0's ndcg@3: 0.611511	valid_0's ndcg@5: 0.649081
[7]	valid_0's ndcg@1: 0.550033	valid_0's ndcg@3: 0.611753	valid_0's ndcg@5: 0.649307
[8]	valid_0's ndcg@1: 0.550236	valid_0's ndcg@3: 0.611599	valid_0's ndcg@5: 0.649597
[9]	valid_0's ndcg@1: 0.550418	valid_0's ndcg@3: 0.611916	valid_0's ndcg@5: 0.650082
[10]	valid_0's ndcg@1: 0.550458	valid_0's ndcg@3: 0.611909	valid_0's ndcg@5: 0.650095
(1135523, 4)
   gid impression  clicked                 prob
0  472      76028        0  0.32929795819497326
1  472      70388        0  0.11374295278224102
2  472    1336053        0  0.03926587118083777
3  472     317126        0 -0.08821200130791787
4  472      79506        0 -0.04294223637748899

Local mrr: 0.5364598305788802
[{'nDCG@1': 0.4003073376872839}, {'nDCG@3': 0.509373241347707}, {'nDCG@5': 0.551707831104393}]
                           feature  importance
90    elapsed_time_between_is_last          50
12                      price_norm          31
89                         is_last          27
13                 price_norm_rank          26
17                 co_pos_min_diff          21
180                   elapsed_time          19
19         clickouted_pos_max_diff          18
11                        pos_rate          18
82                    precnt_ratio          16
83   interaction_item_rating_ratio          12
pipeline elapsed_time: 1.1443207128180397 [hour]
finished
```

# Getting a submission with full training dataset
You can make the submission through loading data, pre-precessing, feature-engineering, local-validation, and prediction, you put into this command as follow 
```
$ python run.py --all 
```   

