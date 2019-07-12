# s2s1osaka
We archived 13th out of 1564 sign-uped teams in RecSys Challenge 2019 with Public score 0.673804 and Final score 0.671117. This publication of code is to build the solution we came up with. Also, our team consists of `yasuo.yamamoto.jp@gmail.com` and `axel.x12x@gmail.com`.  

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
... ... Awareness
... ... BySession
... ... ByItem
... ... ByLocation
... ... JustBeforeClickout
... set interactions between features
... ... Polinomials
... target variable
... ... TargetVariable
... local validation
[LightGBM] [Warning] Find whitespaces in feature_names, replace with underlines
[LightGBM] [Info] Total Bins 13459
[LightGBM] [Info] Number of data: 2091182, number of used features: 186
[LightGBM] [Warning] Find whitespaces in feature_names, replace with underlines
[1]	valid_0's ndcg@1: 0.528126	valid_0's ndcg@3: 0.600165	valid_0's ndcg@5: 0.638517
[2]	valid_0's ndcg@1: 0.543978	valid_0's ndcg@3: 0.608781	valid_0's ndcg@5: 0.645757
[3]	valid_0's ndcg@1: 0.546102	valid_0's ndcg@3: 0.609904	valid_0's ndcg@5: 0.647168
[4]	valid_0's ndcg@1: 0.546587	valid_0's ndcg@3: 0.610802	valid_0's ndcg@5: 0.64787
[5]	valid_0's ndcg@1: 0.546789	valid_0's ndcg@3: 0.610886	valid_0's ndcg@5: 0.648317
[6]	valid_0's ndcg@1: 0.547375	valid_0's ndcg@3: 0.611325	valid_0's ndcg@5: 0.648644
[7]	valid_0's ndcg@1: 0.547557	valid_0's ndcg@3: 0.611425	valid_0's ndcg@5: 0.648751
[8]	valid_0's ndcg@1: 0.54778	valid_0's ndcg@3: 0.61178	valid_0's ndcg@5: 0.649129
[9]	valid_0's ndcg@1: 0.547982	valid_0's ndcg@3: 0.611995	valid_0's ndcg@5: 0.649254
[10]	valid_0's ndcg@1: 0.548063	valid_0's ndcg@3: 0.612354	valid_0's ndcg@5: 0.649202
(1134038, 4)
   gid impression  clicked                  prob
0  434    2269838        0   0.19202540412714303
1  434     211041        0  -0.01283260825256696
2  434    1331323        0  -0.04665281549427339
3  434    1439759        0   0.01273867109885182
4  434      54152        0 -0.004749033911652081

Local mrr: 0.5356470557472142
elapsed_time: 1.1241914156410429 [hour]
finished
```

# Getting a submission with full training dataset
You can make the submission through loading data, pre-precessing, feature-engineering, local-validation, and prediction, you put into this command as follow 
```
$ python run.py --all 
```   

