import pandas as pd
import numpy as np
import datetime
from modeling import target_dtype, parse_dates
from modeling import get_target_cols, get_id_cols
from modeling import lgbm_params, _lgbm_params
from modeling import num_boost_round, _num_boost_round
from modeling.metrics import calc_mrr
import lightgbm as lgb

class Validation(object):
    @classmethod
    def get_pred_df(cls, X_TR, scope="--check"):
        target_cols = get_target_cols()
        id_cols = get_id_cols()

        # datetime of splitting into train and test
        split_dt = datetime.datetime.strptime("2018-11-05 09:00:00", '%Y-%m-%d %H:%M:%S')

        # split train and test
        X_train = X_TR[X_TR.timestamp_dt < split_dt]
        X_test = X_TR[X_TR.timestamp_dt >= split_dt]

        # set groups for rank learning
        Q_train = X_train[['session_id', 'gid']].groupby('session_id').count().reset_index()
        Q_train = Q_train.rename(columns={'gid': 'query'})
        Q_test = X_test[['session_id', 'gid']].groupby('session_id').count().reset_index()
        Q_test = Q_test.rename(columns={'gid': 'query'})

        # shrink by target_cols
        y_train = X_train[["clicked"]]
        X_train = X_train[target_cols + id_cols]
        y_test = X_test[["clicked"]]
        X_test = X_test[target_cols + id_cols]

        # modeling
        lgb_train = lgb.Dataset(X_train[target_cols], y_train, group=Q_train['query'])
        lgb_eval = lgb.Dataset(X_test[target_cols], y_test, group=Q_test['query'], reference=lgb_train)
        if (scope=="--check"):
            model = lgb.train(_lgbm_params
                              , lgb_train
                              , num_boost_round=_num_boost_round
                              , valid_sets=lgb_eval)
        else:
            model = lgb.train(lgbm_params
                              , lgb_train
                              , num_boost_round=num_boost_round
                              , valid_sets=lgb_eval)

        # calc mrr
        y_pred = model.predict(X_test[target_cols], num_iteration=model.best_iteration)
        y_pred_df = pd.DataFrame({"gid": X_test["gid"].values
                                     , "impression": X_test["impression"].values
                                     , "clicked": y_test["clicked"].values
                                     , "prob": y_pred})
        if (scope!="--check"):
            # print feature importance
            fti = model.feature_importance()
            fti_df = pd.DataFrame({"feature": target_cols, "importance": fti})
            print(fti_df.sort_values("importance", ascending=False))

        # display(y_pred_df.head())
        return (y_pred_df, np.mean(y_pred_df.groupby("gid").apply(lambda x: calc_mrr(x))), model)
