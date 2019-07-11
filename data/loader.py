import pandas as pd
import numpy as np
import datetime
import random

class Dataset(object):
    @classmethod
    def load(cls, path="./", scope="--check"):

        dataset = {}
        # - - - - - - - - - -
        if (scope == "--check"):
            n_train = 15932993
            s_train = 1593299
            skip_train = sorted(random.sample(range(n_train), n_train - s_train))
            skip_train = skip_train[1:len(skip_train)]
            n_test = 3782336
            s_test = 378233
            skip_test = sorted(random.sample(range(n_test), n_test - s_test))
            skip_test = skip_test[1:len(skip_test)]
            train_df = pd.read_csv(path + "train.csv", skiprows=skip_train)
            test_df = pd.read_csv(path + "test.csv", skiprows=skip_test)
        else:
            train_df = pd.read_csv(path + "train.csv")
            test_df = pd.read_csv(path + "test.csv")
        # - - - - - - - - - -
        dataset["submission_df"] = pd.read_csv(path + "submission_popular.csv")
        dataset["imeta_df"] = pd.read_csv(path + "item_metadata.csv")

        # convert float to string for reference
        train_df['reference'] = train_df['reference'].astype(str)
        test_df['reference'] = test_df['reference'].astype(str)

        # set is_train flg
        train_df["is_train"] = 1
        test_df["is_train"] = 0

        # set target records
        train_df.index = [np.arange(0, train_df.shape[0], 1)]
        test_df.index = [np.arange(train_df.shape[0], train_df.shape[0] + test_df.shape[0], 1)]
        train_df["idx"] = train_df.index
        test_df["idx"] = test_df.index
        train_df["is_y"] = 0
        test_df["is_y"] = 0
        y_df1 = train_df[train_df.action_type == "clickout item"].groupby("session_id").last().reset_index()
        y_df2 = test_df[(test_df.action_type == "clickout item") & (test_df.reference == "nan")]
        train_df.loc[train_df.idx.isin(y_df1.idx), "is_y"] = 1
        test_df.loc[test_df.idx.isin(y_df2.idx), "is_y"] = 1

        # because of there are duplicate session id in both of train and test, it's renaming session_id in train.
        train_df["session_id"] = "t" + train_df["session_id"]
        all_df = pd.concat([train_df, test_df])

        # all_df = train_df # null check
        all_df["timestamp_dt"] = all_df["timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(np.float(x)))
        dataset["all_df"] = all_df
        del train_df
        del test_df
        del y_df1
        del y_df2
        return dataset
