from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import collections
import itertools
from abc import ABCMeta, abstractmethod


class IAdditionalDataset(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def create(self, dataset):
        pass


class CityForSessionStep(IAdditionalDataset):
    @classmethod
    def create(cls, dataset):
        print("... ... CityForSessionStep")
        all_df = dataset["all_df"]
        city_all_df = all_df[["city"]].copy()
        city_all_df = city_all_df[~city_all_df.duplicated()]
        city_dict = {str(c): str(i) for i, c in enumerate(city_all_df["city"].tolist())}
        dataset["city_dict"] = city_dict

        # for all_df
        all_df["city_id"] = all_df[["city"]].apply(lambda x: city_dict[str(x.city)], axis=1)
        all_df["_session_id"] = all_df["session_id"] + all_df["city_id"]
        del city_all_df

        step_all_df = all_df[["session_id", "step", "_session_id"]].copy()
        step_all_df = step_all_df[["_session_id"]].groupby("_session_id").cumcount().reset_index()
        step_all_df.columns = ["_", "_step"]
        step_all_df["_step"] += 1
        step_all_df.index = all_df.index
        all_df = pd.concat([all_df, step_all_df[["_step"]]], axis=1)
        del step_all_df

        all_df["country_name"] = all_df[["city"]].apply(lambda x: x.city.split(",")[-1].strip(), axis=1)
        dataset["all_df"] = all_df


class ItemPropsVector(IAdditionalDataset):
    @classmethod
    def create(cls, dataset):
        print("... ... ItemPropsVector")
        imeta_df = dataset["imeta_df"]
        imeta_df["property"] = imeta_df[["properties"]].apply(lambda x: str(x.properties).split("|"), axis=1)
        expanded_imeta_df = imeta_df.set_index('item_id').property.apply(pd.Series).stack().reset_index(level=0).rename(
            columns={0: 'property'})
        onehot_ratings = pd.get_dummies(expanded_imeta_df, columns=['property'])
        item_props = onehot_ratings.groupby('item_id').sum().reset_index()
        cols = [col.replace("property_", "p") for col in list(item_props.columns)]
        item_props.columns = ["item_id"] + cols[1:len(cols)]
        item_props["item_id"] = item_props["item_id"].astype(str)
        del expanded_imeta_df
        del onehot_ratings
        dataset["item_props"] = item_props
        ItemPropsVector._calc_coverate(dataset)
        ItemPropsVector._create_dense(dataset)

    @classmethod
    def _calc_coverate(cls, dataset):
        item_props = dataset["item_props"]
        sum_item_props = item_props.sum()
        sum_item_props_df = pd.DataFrame(sum_item_props)

        sum_item_props_df = sum_item_props_df.reset_index()
        sum_item_props_df.columns = ["prop", "cnt"]
        sum_item_props_df = sum_item_props_df.drop(0)
        sum_item_props_df["coverage"] = sum_item_props_df["cnt"] / item_props.shape[0]
        dataset["sum_item_props_df"] = sum_item_props_df

    @classmethod
    def _create_dense(cls, dataset):
        item_props = dataset["item_props"]
        n_components = 10
        svd = TruncatedSVD(n_components=n_components, n_iter=100, random_state=1234)
        item_props_cols = list(item_props.columns)
        item_props_svd = pd.DataFrame(svd.fit_transform(item_props[item_props_cols[1:len(item_props_cols)]]))
        item_props_svd.columns = ["prop_svd_" + str(i + 1) for i in range(n_components)]
        item_props_svd = pd.concat([item_props[["item_id"]], item_props_svd], axis=1)
        dataset["item_props_svd"] = item_props_svd

class View2viewCounter(IAdditionalDataset):
    @classmethod
    def create(cls, dataset):
        print("... ... View2viewCounter")
        # v2v
        atype_long_names = ["clickout item"]
        all_df = dataset["all_df"]
        v2v_df = all_df[["action_type", "session_id", "reference", "is_y"]].copy()

        v2v_df = v2v_df[v2v_df.action_type.isin(atype_long_names)]

        v2v_df = v2v_df[v2v_df.is_y == 0]
        v2v_df = v2v_df[v2v_df.reference != "nan"]

        v2v_df = v2v_df[["session_id", "reference"]]
        v2v_df = v2v_df.groupby("session_id").apply(lambda x: list(set(list(x.reference)))).reset_index()
        v2v_df.columns = ["session_id", "references"]
        v2v_df["lenref"] = v2v_df.apply(lambda x: len(x.references), axis=1)
        v2v_df = v2v_df[v2v_df.lenref > 1]

        v2v_dict = {}
        v2v_dict_inv = {}
        def create_v2v_dict(references):
            for kv in list(itertools.combinations(references, 2)):
                if kv[0] in v2v_dict:
                    v2v_dict[kv[0]].append(kv[1])
                else:
                    v2v_dict[kv[0]] = [kv[1]]
                if kv[1] in v2v_dict_inv:
                    v2v_dict_inv[kv[1]].append(kv[0])
                else:
                    v2v_dict_inv[kv[1]] = [kv[0]]

        _ = v2v_df[["references"]].apply(lambda x: create_v2v_dict(x.references), axis=1)

        v2v_counter = {}
        for k in v2v_dict.keys():
            v2v_counter[k] = v2v_dict[k]

        for k in v2v_dict_inv.keys():
            if k in v2v_counter:
                v2v_counter[k] = v2v_counter[k] + v2v_dict_inv[k]
            else:
                v2v_counter[k] = v2v_dict_inv[k]

        for k in v2v_counter.keys():
            v2v_counter[k] = collections.Counter(v2v_counter[k])

        dataset["v2v_counter"] = v2v_counter

class BayesLikelihood(IAdditionalDataset):
    @classmethod
    def create(cls, dataset):
        print("... ... BayesLikelihood")
        all_df = dataset["all_df"]

        # extract ref, last_ref - - - - - -
        action_types = ["interaction item rating"
            , "interaction item info"
            , "interaction item image"
            , "interaction item deals"
            , "search for item"
            , "clickout item"]

        bayes_df = all_df[["session_id", "action_type", "reference", "is_y"]].copy()

        bayes_df["last_action_type"] = bayes_df["action_type"].shift(1)
        bayes_df["last_session_id"] = bayes_df["session_id"].shift(1)
        bayes_df["last_reference"] = bayes_df["reference"].shift(1)

        bayes_df = bayes_df[bayes_df.session_id == bayes_df.last_session_id]
        bayes_df = bayes_df[bayes_df.is_y == 0]
        bayes_df = bayes_df[bayes_df.action_type == "clickout item"]
        bayes_df = bayes_df[bayes_df.last_action_type.isin(action_types)]
        bayes_df = bayes_df[["reference", "last_reference", "action_type", "last_action_type"]]

        # calc bayes score - - - - - -
        # calc prior
        prior_df = pd.DataFrame(bayes_df["reference"].value_counts()).reset_index()
        prior_df.columns = ["reference", "prior"]
        prior_df["prior"] = prior_df["prior"].astype(float) / prior_df.shape[0]

        # calc likelihood
        bayes_score_df = bayes_df[["reference", "last_reference"]].groupby(
            ["reference", "last_reference"]).size().reset_index()
        bayes_score_df.columns = ["reference", "last_reference", "a_cnt"]

        b_df = bayes_df[["reference"]].groupby(["reference"]).size().reset_index()
        b_df.columns = ["reference", "b_cnt"]

        bayes_score_df = pd.merge(bayes_score_df, b_df, on="reference", how="left")
        bayes_score_df["likelihood"] = bayes_score_df["a_cnt"].astype(float) / bayes_score_df["b_cnt"].astype(float)
        bayes_score_df = bayes_score_df[["reference", "last_reference", "likelihood"]]

        # calc posterior
        bayes_score_df = pd.merge(bayes_score_df, prior_df, on="reference", how="left")
        bayes_score_df["posterior"] = bayes_score_df["likelihood"] * bayes_score_df["prior"]

        # make keys
        bayes_score_df["rlr"] = bayes_score_df["reference"].astype(str) + bayes_score_df["last_reference"].astype(str)

        # export bayes dict
        bayes_likelihood = {k: v for k, v in zip(bayes_score_df["rlr"].tolist(), bayes_score_df["likelihood"].tolist())}

        del bayes_df
        del bayes_score_df
        dataset["bayes_likelihood"] = bayes_likelihood


class ImpressionScore(IAdditionalDataset):
    @classmethod
    def create(cls, dataset):
        print("... ... ImpressionScore")
        all_df = dataset["all_df"]
        imps_df = all_df[~all_df.impressions.isnull()][["impressions"]].copy()
        imps_df["impid"] = imps_df.index
        imps_df["impression"] = imps_df[["impressions"]].apply(lambda x: str(x.impressions).split("|"), axis=1)
        impscore_df = imps_df[["impid", "impression"]].set_index('impid').impression.apply(
            pd.Series).stack().reset_index(level=0).rename(columns={0: 'impression'})
        imps_cnt_df = imps_df[["impid", "impression"]].set_index('impid').impression.apply(
            lambda x: pd.Series([1. / (p + 1) for p in range(len(x))])).stack().reset_index(level=0).rename(
            columns={0: 'imps_cnt'})
        impscore_df = pd.concat([impscore_df, imps_cnt_df], axis=1)
        impscore_df = impscore_df[["impression", "imps_cnt"]].groupby("impression").apply(
            lambda x: np.mean(x.imps_cnt)).reset_index()
        impscore_df.columns = ["impression", "impsocre"]

        del imps_df
        del imps_cnt_df
        dataset["impscore_df"] = impscore_df