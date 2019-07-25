import pandas as pd
pd.options.display.float_format = '{}'.format
pd.options.display.max_rows = 25
pd.options.display.max_columns = 500
pd.set_option("display.max_colwidth", 240)
import numpy as np
import datetime
import random
import time
import random
import gc
import sys

from data.loader import Dataset
from data.preprocessing import CityForSessionStep
from data.preprocessing import ItemPropsVector
from data.preprocessing import View2viewCounter
from data.preprocessing import BayesLikelihood
from data.preprocessing import ImpressionScore

from data.features import Durations
from data.features import JustClickout
from data.features import Record2Impression
from data.features import EncodingForCategories
from data.features import BySession
from data.features import ByItem
from data.features import ByLocation
from data.features import JustBeforeClickout
from data.features import DecisionMakingProcess
from data.features import Polinomials
from data.features import TargetVariable

from modeling.local import Validation
from modeling.public import Prediction
from modeling.public import Submission
from modeling import get_target_cols

from modeling.metrics import calc_ndcg


def pipeline(scope="check"):
    start = time.time()
    print("starting")
    sys.stdout.flush()

    # loading dataset
    print("... loading dataset")
    sys.stdout.flush()
    dataset = Dataset.load(path="./data_v2/", scope=scope)

    # preprocessing
    print("... preprocessing")
    sys.stdout.flush()
    CityForSessionStep.create(dataset)
    ItemPropsVector.create(dataset)
    View2viewCounter.create(dataset)
    BayesLikelihood.create(dataset)
    ImpressionScore.create(dataset)
    gc.collect()

    # create rows for training as X
    print("... create rows for training as X")
    sys.stdout.flush()
    extract_cols = ["user_id"
        , "_session_id"
        , "session_id"
        , "timestamp"
        , "timestamp_dt"
        , "step"
        , "_step"
        , "reference"
        , "platform"
        , "city"
        , "current_filters"
        , "device"
        , "country_name"
        , "is_train"
        , "is_y"]
    X = dataset["all_df"][dataset["all_df"].is_y == 1].copy()
    X = X[extract_cols + ["impressions", "prices"]]
    gc.collect()

    # set dummy variables for device
    print("... set dummy variables for device")
    sys.stdout.flush()
    device_df = pd.get_dummies(X[["device"]], columns=['device'])
    device_df.columns = ["desktop", "mobile", "tablet"]
    X = pd.concat([X, device_df[["desktop", "mobile"]]], axis=1)
    extract_cols = extract_cols + ["desktop", "mobile"]
    del device_df

    # feature engineering to X
    print("... feature engineering to X")
    sys.stdout.flush()
    X, extract_cols = Durations.set(X, extract_cols, dataset)
    X, extract_cols = JustClickout.set(X, extract_cols)
    gc.collect()

    # expanding X
    print("... expanding X")
    sys.stdout.flush()
    X, extract_cols = Record2Impression.expand(X, extract_cols, dataset)
    gc.collect()

    # feature engineering to expanded X
    print("... feature engineering to expanded X")
    sys.stdout.flush()
    X = EncodingForCategories.to_prob(X, dataset)
    sys.stdout.flush()
    X = DecisionMakingProcess.detect(X, dataset)
    sys.stdout.flush()
    X = BySession.set(X, dataset)
    sys.stdout.flush()
    X = ByItem.set(X, dataset)
    sys.stdout.flush()
    X = ByLocation.set(X, dataset)
    sys.stdout.flush()
    X = JustBeforeClickout.set(X, dataset)
    sys.stdout.flush()
    gc.collect()

    # set interactions between features
    print("... set interactions between features")
    sys.stdout.flush()
    X = Polinomials.set(X)

    # target variable
    print("... target variable")
    sys.stdout.flush()
    X = TargetVariable.set(X)

    # local validation
    print("... local validation")
    sys.stdout.flush()
    X_TR = X[X.is_train == 1]
    y_pred_df, mrr, model = Validation.get_pred_df(X_TR, scope=scope)
    gc.collect()
    print(y_pred_df.shape)
    print(y_pred_df.head())
    print("")
    print("Local mrr: {}".format(mrr))
    print(calc_ndcg(y_pred_df))
    fti = model.feature_importance()
    fti_df = pd.DataFrame({"feature": get_target_cols(), "importance": fti})
    print(fti_df.sort_values("importance", ascending=False).head(10))

    if (scope!="--check"):
        # create submission
        print("... create submission")
        sys.stdout.flush()
        submission_df = dataset["submission_df"].copy()
        del dataset
        gc.collect()
        X_TE = X[X.is_train == 0]
        y_pred_df = Prediction.get_pred_df(X_TR, X_TE)
        IDCOLS_DF = X[["gid", "user_id", "session_id", "step"]].copy()
        IDCOLS_DF = IDCOLS_DF[~IDCOLS_DF.duplicated()]
        sub_df = Submission.get_sub_df(y_pred_df, IDCOLS_DF, submission_df)
        prefix = "{}_{:0=4}".format(datetime.datetime.now().strftime("%Y%m%d")
                                    , random.randint(0, 100))
        sub_csv = "./subs/" + prefix + "_submission.csv"
        sub_df.to_csv(sub_csv, header=True, index=False, sep=',')
        gc.collect()
        print(sub_csv)

    elapsed_time = time.time() - start
    print("pipeline elapsed_time: {0}".format(elapsed_time/60/60) + " [hour]")
    print("finished")

if __name__ == "__main__":
    args = sys.argv
    if 2 <= len(args):
        scope = str(args[1])
        if (scope == "--check") or (scope == "--all"):
            pipeline(scope=scope)
        else:
            print("Arguments are too short.")
            print("Usage: python run.py [--check|--all]")
            print("--check: in order to do code walkthrough without making submission")
            print("--all: to do all pipeline with full dataset")
    else:
        print("Arguments are too short.")
        print("Usage: python run.py [--check|--all]")
        print("--check: in order to do code walkthrough without making submission")
        print("--all: to do all pipeline with full dataset")
