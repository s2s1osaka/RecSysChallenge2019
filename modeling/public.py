import pandas as pd
import numpy as np
import datetime
from modeling import get_target_cols
from modeling import lgbm_params, num_boost_round
import lightgbm as lgb


class Prediction(object):
    @classmethod
    def get_pred_df(cls, X_TR, X_TE):
        target_cols = get_target_cols()

        # full training
        Q_TR = X_TR[['session_id', 'gid']].groupby('session_id').count().reset_index()
        Q_TR = Q_TR.rename(columns={'gid': 'query'})
        lgb_train = lgb.Dataset(X_TR[target_cols], X_TR["clicked"], group=Q_TR['query'])
        model = lgb.train(lgbm_params
                          , lgb_train
                          , num_boost_round=num_boost_round
                          , verbose_eval=1)

        # prediction
        y_pred = model.predict(X_TE[target_cols], num_iteration=model.best_iteration)
        y_pred_df = pd.DataFrame({"gid": X_TE["gid"].values
                                     , "impression": X_TE["impression"].values
                                     , "prob": y_pred})
        return y_pred_df


class Submission(object):
    @classmethod
    def get_sub_df(cls, y_pred_df, IDCOLS_DF, dataset):
        submission_df = dataset["submission_df"]

        # rank normalization
        y_pred_df["prob"] = y_pred_df["prob"].rank(ascending=True)
        y_pred_df["prob"] = y_pred_df["prob"].astype(int)

        # create submission
        y_pred_df["impression"] = y_pred_df["impression"].astype(str)
        def create_sub(x):
            impressions = list(x.impression)
            probs = list(x.prob)
            imps_probs = [[i, p] for i, p in zip(impressions, probs)]
            imps_probs.sort(key=lambda z: z[1])
            imps = [imps_prob[0] for imps_prob in imps_probs]
            return " ".join(imps)

        mysub_df = y_pred_df.groupby("gid").apply(lambda x: create_sub(x)).reset_index()
        mysub_df = pd.merge(mysub_df, IDCOLS_DF, on="gid", how="left")
        mysub_df.columns = ["gid", "item_recommendations", "user_id", "session_id", "step"]
        mysub_df = mysub_df[["user_id", "session_id", "step", "item_recommendations"]]
        mysub_df.columns = ["user_id", "session_id", "step", "item_recommendations_sub"]
        sub_df = pd.merge(submission_df, mysub_df, on=["user_id", "session_id", "step"], how="left")
        sub_df = sub_df[["user_id", "session_id", "timestamp", "step", "item_recommendations_sub"]]
        sub_df.columns = ["user_id", "session_id", "timestamp", "step", "item_recommendations"]
        return sub_df
