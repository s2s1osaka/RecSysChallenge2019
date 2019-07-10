import pandas as pd
import numpy as np
import datetime

class Durations(object):
    @classmethod
    def set(cls, X, extract_cols, dataset):
        print("... ... Durations")
        all_df = dataset["all_df"]

        # duration from first action to clickout
        dffac_df = all_df[["session_id", "timestamp", "timestamp_dt"]].groupby(
            "session_id").first().reset_index()
        dffac_df = dffac_df[["session_id", "timestamp_dt"]]
        dffac_df.columns = ["session_id", "first_timestamp_dt"]
        X = pd.merge(X, dffac_df, on="session_id", how="left")
        X["session_duration"] = X.apply(lambda x: (x.timestamp_dt - x.first_timestamp_dt).seconds, axis=1)
        extract_cols = extract_cols + ["session_duration"]
        del dffac_df

        # duration from last distination to clickout
        dflsc_df = all_df[["session_id", "_session_id", "timestamp", "timestamp_dt"]].groupby(
            "_session_id").first().reset_index()
        dflsc_df = dflsc_df[dflsc_df._session_id.isin(X._session_id)]
        dflsc_df = dflsc_df[["session_id", "timestamp_dt"]]
        dflsc_df.columns = ["session_id", "step_first_timestamp_dt"]
        X = pd.merge(X, dflsc_df, on="session_id", how="left")
        X["step_duration"] = X.apply(lambda x: (x.timestamp_dt - x.step_first_timestamp_dt).seconds, axis=1)
        extract_cols = extract_cols + ["step_duration"]
        del dflsc_df

        return (X, extract_cols)


class JustClickout(object):
    @classmethod
    def set(cls, X, extract_cols):
        print("... ... JustClickout")
        # append current fillters
        def get_cf_features(x):
            sbp = 1 if "Sort by Price" in x.current_filters else 0
            sbd = 1 if "Sort By Distance" in x.current_filters else 0
            sbr = 1 if "Sort By Rating" in x.current_filters else 0
            fod = 1 if "Focus on Distance" in x.current_filters else 0
            fsr = 1 if "Focus on Rating" in x.current_filters else 0
            bev = 1 if "Best Value" in x.current_filters else 0
            return pd.Series({'cf_sbp': sbp
                                 , 'cf_sbd': sbd
                                 , 'cf_sbr': sbr
                                 , 'cf_fod': fod
                                 , 'cf_fsr': fsr
                                 , 'cf_bev': bev})

        X["current_filters"] = X["current_filters"].fillna("")
        curf_df = X[["current_filters"]].apply(lambda x: get_cf_features(x), axis=1)
        X = pd.concat([X, curf_df], axis=1)
        extract_cols = extract_cols + list(curf_df.columns)
        del curf_df

        return (X, extract_cols)


class JustBeforeClickout(object):
    @classmethod
    def set(cls, X, dataset):
        print("... ... JustBeforeClickout")
        all_df = dataset["all_df"]

        # last action_type
        lasttype_df = all_df[["session_id", "action_type", "is_y"]].copy()
        lasttype_df["lat"] = lasttype_df["action_type"].shift(1)
        lasttype_df["last_session_id"] = lasttype_df["session_id"].shift(1)

        lasttype_df = lasttype_df[lasttype_df.is_y == 1]

        lasttype_df = lasttype_df[lasttype_df.session_id == lasttype_df.last_session_id]
        lasttype_df = lasttype_df[["session_id", "lat"]]
        onehot_lat = pd.get_dummies(lasttype_df, columns=['lat'])
        X = pd.merge(X, onehot_lat, on="session_id", how="left")
        lat_cols = list(onehot_lat.columns)
        lat_cols.remove("session_id")
        for lat_col in lat_cols:
            X[lat_col] = X[lat_col].fillna(0)
        del lasttype_df
        del onehot_lat


class Record2Impression(object):
    @classmethod
    def expand(cls, X, extract_cols):
        print("... ... Record2Impression")
        # create expanded
        X = X.reset_index()
        X["gid"] = X.index
        X["n_imps"] = X[["impressions"]].apply(lambda x: len(str(x.impressions).split("|")), axis=1)
        X["price_mean"] = X[["prices"]].apply(lambda x: np.mean(np.array(str(x.prices).split("|")).astype(int)), axis=1)
        X["price_std"] = X[["prices"]].apply(lambda x: np.std(np.array(str(x.prices).split("|")).astype(int)), axis=1)
        X["impression"] = X[["impressions"]].apply(lambda x: str(x.impressions).split("|"), axis=1)
        X["price"] = X[["prices"]].apply(lambda x: str(x.prices).split("|"), axis=1)
        X_impression = X[["gid", "impression"]].set_index('gid').impression.apply(pd.Series).stack().reset_index(
            level=0).rename(columns={0: 'impression'})
        X_price = X[["gid", "price"]].set_index('gid').price.apply(pd.Series).stack().reset_index(level=0).rename(
            columns={0: 'price'})
        X_position = X[["gid", "impression"]].set_index('gid').impression.apply(
            lambda x: pd.Series(range(len(x)))).stack().reset_index(level=0).rename(columns={0: 'position'})
        X_expanded = pd.concat([X_impression, X_price], axis=1)
        X_expanded = pd.concat([X_expanded, X_position], axis=1)
        X_expanded.columns = ["gid", "impression", "gid2", "price", "gid3", "position"]
        X_expanded = X_expanded[["gid", "impression", "price", "position"]]

        # join expaned
        X = pd.merge(X_expanded, X[["gid", "n_imps", "price_mean", "price_std"] + extract_cols], on="gid", how="left")

        # to normalize position and price
        X["pos_rate"] = X["position"] / X["n_imps"]
        X["pos"] = X["position"] + 1
        X["price_norm"] = (X["price"].astype(float) - X["price_mean"].astype(float)) / X["price_std"].astype(float)

        # join price_norm rank
        pnorm_rank_df = X[["session_id", "price_norm"]].copy()
        pnorm_rank_df = pnorm_rank_df[["session_id", "price_norm"]].groupby("session_id").rank(ascending=False)
        pnorm_rank_df.columns = ["price_norm_rank"]
        X = pd.concat([X, pnorm_rank_df], axis=1)
        del pnorm_rank_df

        # calc discount rate
        X["price"] = X["price"].astype(float)
        prices_df = X[["impression", "price"]].groupby("impression").agg({'price': np.mean}).reset_index()
        prices_df.columns = ["impression", "item_price_mean"]
        X = pd.merge(X, prices_df, on="impression", how="left")
        X["discount_rate"] = X["price"] / X["item_price_mean"]
        del prices_df

        return (X, extract_cols)


class Perception(object):
    @classmethod
    def detect(cls, X, dataset):
        print("... ... Perceptions")
        all_df = dataset["all_df"]

        # join pos stats"
        copos_df = all_df[all_df.action_type == "clickout item"][
            ["session_id", "reference", "impressions", "is_y"]].copy()
        copos_df = copos_df[copos_df.is_y == 0]
        copos_df["impression"] = copos_df[["impressions"]].apply(lambda x: str(x.impressions).split("|"), axis=1)
        copos_df["co_pos"] = copos_df[["impression", "reference"]].apply(
            lambda x: x.impression.index(x.reference) + 1 if x.reference in x.impression else 1, axis=1)
        copos_df_stats = copos_df[["session_id", "co_pos"]].groupby("session_id").agg(
            {'co_pos': [np.min, np.max, np.mean]}).reset_index()
        copos_df_stats.columns = ["session_id", "co_pos_min", "co_pos_max", "co_pos_mean"]
        X = pd.merge(X, copos_df_stats, on="session_id", how="left")
        X["co_pos_min"] = X["co_pos_min"].fillna(1)
        X["co_pos_mean"] = X["co_pos_mean"].fillna(1)
        X["co_pos_max"] = X["co_pos_max"].fillna(1)
        X["co_pos_min_diff"] = X["pos"] - X["co_pos_min"]
        X["co_pos_mean_diff"] = X["pos"] - X["co_pos_mean"]
        X["co_pos_max_diff"] = X["co_pos_max"] - X["pos"]
        del copos_df
        del copos_df_stats

        # is_last and is_last_elapsed_time
        action_types = ["interaction item rating"
            , "interaction item info"
            , "interaction item image"
            , "interaction item deals"
            , "search for item"
            , "clickout item"]
        lastref_df = all_df[["session_id", "action_type", "reference", "timestamp", "is_y"]].copy()
        lastref_df["is_target"] = 0
        lastref_df.loc[lastref_df.is_y == 1, "is_target"] = 1
        lastref_df = lastref_df[lastref_df.action_type.isin(action_types)]
        lastref_df["last_session_id"] = lastref_df["session_id"].shift(1)
        lastref_df["last_reference"] = lastref_df["reference"].shift(1)
        lastref_df["last_timestamp"] = lastref_df["timestamp"].shift(1)
        lastref_df = lastref_df[lastref_df.session_id == lastref_df.last_session_id]
        lastref_df = lastref_df[lastref_df.is_target == 1][["session_id", "last_reference", "last_timestamp"]]
        X = pd.merge(X, lastref_df, on="session_id", how="left")
        X[["last_reference"]] = X[["last_reference"]].fillna("-1")
        X[["last_timestamp"]] = X[["last_timestamp"]].fillna(-1)
        X["is_last"] = X[["impression", "last_reference"]].apply(lambda x: 1 if x.impression == x.last_reference else 0,
                                                                 axis=1)
        X["last_elapsed_time"] = X[["impression", "last_reference", "timestamp", "last_timestamp"]].apply(
            lambda x: int(x.timestamp) - int(x.last_timestamp) if x.impression == x.last_reference else np.nan, axis=1)
        lastdur_df = X[["session_id", "last_elapsed_time"]].copy()
        lastdur_df = lastdur_df.dropna(axis=0, how='any')
        X.drop("last_elapsed_time", axis=1, inplace=True)
        X = pd.merge(X, lastdur_df, on="session_id", how="left")
        del lastref_df
        del lastdur_df

        # join is_last_last
        lastref_df = all_df[["session_id", "action_type", "reference", "is_y"]].copy()
        lastref_df["last_last_session_id"] = lastref_df["session_id"].shift(2)
        lastref_df["last_last_reference"] = lastref_df["reference"].shift(2)
        lastref_df = lastref_df[lastref_df.is_y == 1]
        lastref_df = lastref_df[lastref_df.session_id == lastref_df.last_last_session_id]
        lastref_df = lastref_df[["session_id", "last_last_reference"]]
        lastref_df = lastref_df[~lastref_df.duplicated()]
        X = pd.merge(X, lastref_df, on="session_id", how="left")
        X[["last_last_reference"]] = X[["last_last_reference"]].fillna("-1")
        X["is_last_last"] = X[["impression", "last_last_reference"]].apply(
            lambda x: 1 if x.impression == x.last_last_reference else 0, axis=1)
        del lastref_df

        # elapsed next mean by item "it's kind of a future information."
        action_types = ["interaction item rating"
            , "interaction item info"
            , "interaction item image"
            , "interaction item deals"
            , "search for item"
            , "clickout item"]
        isnext_df = all_df[["session_id", "action_type", "reference", "timestamp", "is_y"]].copy()
        isnext_df["next_session_id"] = isnext_df["session_id"].shift(-1)
        isnext_df["next_timestamp"] = isnext_df["timestamp"].shift(-1)
        isnext_df = isnext_df[isnext_df.session_id == isnext_df.next_session_id]
        isnext_df["elapsed_next"] = isnext_df["next_timestamp"] - isnext_df["timestamp"]
        isnext_df = isnext_df[isnext_df.action_type.isin(action_types)]
        isnext_df = isnext_df[isnext_df.is_y == 0]
        isnext_gp_df = isnext_df[["reference", "elapsed_next"]].groupby("reference").agg(
            {"elapsed_next": np.mean}).reset_index()
        isnext_gp_df.columns = ["impression", "next_elapsed_time"]
        X = pd.merge(X, isnext_gp_df, on="impression", how="left")
        del isnext_gp_df

        isnext_gp_df = isnext_df[isnext_df.action_type == "clickout item"][["reference", "elapsed_next"]].groupby(
            "reference").agg({"elapsed_next": np.mean}).reset_index()
        isnext_gp_df.columns = ["impression", "next_elapsed_time_byco"]
        X = pd.merge(X, isnext_gp_df, on="impression", how="left")
        del isnext_df
        del isnext_gp_df

        # set two above displayed item and five below displayed item
        u_cols = []
        def set_undert_the_clickouted_and_islast(X, target_col, nu=5):
            u_col = target_col + "_u"
            X[u_col] = X["session_id"]
            X.loc[X[target_col] != 1, u_col] = ""
            for u in [_ for _ in range(-2, nu + 1, 1) if _ != 0]:
                new_col = u_col + str(u).replace("-", "p")
                X[new_col] = X[u_col].shift(u)
                X[new_col] = X[new_col].fillna("")
                X.loc[X[new_col] == X["session_id"], new_col] = "1"
                X.loc[X[new_col] != "1", new_col] = 0
                X.loc[X[new_col] == "1", new_col] = 1
                u_cols.append(new_col)
            X.drop(u_col, axis=1, inplace=True)
        set_undert_the_clickouted_and_islast(X, "clickouted", 5)
        set_undert_the_clickouted_and_islast(X, "is_last", 5)

        # sum of number of above displayed item
        u_coted_cols = [col for col in u_cols if "clickouted" in col]
        u_islast_col = [col for col in u_cols if "is_last" in col]
        X["clickouted_sum"] = X[u_coted_cols].sum(axis=1)
        X["is_last_sum"] = X[u_islast_col].sum(axis=1)

        # step_elapsed_mean which represents velocity of user activities.
        selapsed_df = all_df[["session_id", "step", "timestamp", "timestamp_dt", "action_type", "reference"]].copy()
        selapsed_df["pre_timestamp"] = selapsed_df["timestamp"].shift(1)
        selapsed_df["pre_timestamp_dt"] = selapsed_df["timestamp_dt"].shift(1)
        selapsed_df["pre_session_id"] = selapsed_df["session_id"].shift(1)
        selapsed_df = selapsed_df[selapsed_df.session_id == selapsed_df.pre_session_id]
        selapsed_df["elapsed"] = selapsed_df["timestamp"] - selapsed_df["pre_timestamp"]
        selapsed_df = selapsed_df[["session_id", "elapsed"]]
        selapsed_df = selapsed_df[selapsed_df.elapsed.notna()]
        selapsed_df = selapsed_df[selapsed_df.elapsed > 0]
        selapsed_df = selapsed_df.groupby("session_id").agg({"elapsed": np.mean}).reset_index()
        selapsed_df.columns = ["session_id", "step_elapsed_mean"]
        X = pd.merge(X, selapsed_df, on="session_id", how="left")
        del selapsed_df

        return X


class ByItem(object):
    @classmethod
    def set(cls, X, dataset):
        print("... ... ByItem")
        all_df = dataset["all_df"]

        # imps score
        impscore_df = dataset["impscore_df"]
        item_props = dataset["item_props"]
        X = pd.merge(X, impscore_df, on="impression", how="left")
        X["impsocre"] = X["impsocre"].fillna(0)

        # append some important props and other props with over 0.2 coverage
        sum_item_props_df = dataset["sum_item_props_df"]
        prop_cols = ["pGood Rating"
            , "pVery Good Rating"
            , "pExcellent Rating"
            , "pSatisfactory Rating"
            , "p1 Star"
            , "p2 Star"
            , "p3 Star"
            , "p4 Star"
            , "p5 Star"
            , "pBusiness Centre"
            , "pBusiness Hotel"
            , "pConference Rooms"]
        c02over_prop_cols = sum_item_props_df[sum_item_props_df.coverage >= 0.2]["prop"].tolist()
        prop_cols = prop_cols + c02over_prop_cols
        prop_cols = list(set(prop_cols))
        X = pd.merge(X, item_props[["item_id"] + prop_cols], left_on="impression", right_on="item_id", how="left")
        X[prop_cols] = X[prop_cols].fillna(0)

        # append item svd n_components=10
        item_props_svd = dataset["item_props_svd"]
        prop_svd_cols = list(item_props_svd.columns)
        prop_svd_cols.remove("item_id")
        X = pd.merge(X, item_props_svd, left_on="impression", right_on="item_id", how="left")
        X[prop_svd_cols] = X[prop_svd_cols].fillna(0)

        # price norm by item rating prop
        X["r6"] = 0
        X["r7"] = 0
        X["r8"] = 0
        X["r9"] = 0
        X.loc[X["pSatisfactory Rating"] == 1, "r6"] = 6
        X.loc[X["pGood Rating"] == 1, "r7"] = 7
        X.loc[X["pVery Good Rating"] == 1, "r8"] = 8
        X.loc[X["pExcellent Rating"] == 1, "r9"] = 9
        X["rating"] = X[["r6", "r7", "r8", "r9"]].apply(
            lambda x: np.mean(np.trim_zeros(np.array([x.r6, x.r7, x.r8, x.r9]))), axis=1)
        X["rating"] = X["rating"].fillna(-1)
        pns_df = X[["session_id", "rating", "price"]].groupby(["session_id", "rating"]).agg(
            {'price': [np.mean, np.std]}).reset_index()
        pns_df.columns = ["session_id", "rating", "r_price_mean", "r_price_std"]
        pns_df["r_price_std"] = pns_df["r_price_std"].fillna(1)
        X = pd.merge(X, pns_df, on=["session_id", "rating"], how="left")
        X["r_price_norm"] = (X["price"].astype(float) - X["r_price_mean"].astype(float)) / X["r_price_std"].astype(
            float)
        del pns_df

        # price norm by star
        X["star"] = -1
        X.loc[X["p1 Star"] == 1, "star"] = 1
        X.loc[X["p2 Star"] == 1, "star"] = 2
        X.loc[X["p3 Star"] == 1, "star"] = 3
        X.loc[X["p4 Star"] == 1, "star"] = 4
        X.loc[X["p5 Star"] == 1, "star"] = 5
        pns_df = X[["session_id", "star", "price"]].groupby(["session_id", "star"]).agg(
            {'price': [np.mean, np.std]}).reset_index()
        pns_df.columns = ["session_id", "star", "s_price_mean", "s_price_std"]
        pns_df["s_price_std"] = pns_df["s_price_std"].fillna(1)
        X = pd.merge(X, pns_df, on=["session_id", "star"], how="left")
        X["s_price_norm"] = (X["price"].astype(float) - X["s_price_mean"].astype(float)) / X["s_price_std"].astype(
            float)
        del pns_df

        # item ctr
        ctrbyitem_df = all_df[all_df.action_type == "clickout item"][["session_id", "reference", "is_y"]].copy()
        ctrbyitem_df = ctrbyitem_df[ctrbyitem_df.is_y == 0]
        ref_df = ctrbyitem_df[["reference"]].groupby(["reference"]).size().reset_index()
        ref_df.columns = ["impression", "rcnt"]
        ref_df["ctrbyitem"] = ref_df["rcnt"].astype(float) / ref_df.shape[0]
        ref_df = ref_df[["impression", "ctrbyitem"]]
        X = pd.merge(X, ref_df, on="impression", how="left")
        X["ctrbyitem"] = X["ctrbyitem"].fillna(0)
        del ctrbyitem_df
        del ref_df

        # item ctr by city
        cr_tmp_df = all_df[all_df.action_type == "clickout item"].copy()
        cr_tmp_df = cr_tmp_df[cr_tmp_df.is_y == 0] # to prevent leakage
        city_df = cr_tmp_df[["city"]].groupby(["city"]).size().reset_index()
        city_df.columns = ["city", "ccnt"]
        cityref_df = cr_tmp_df[["city", "reference"]].groupby(["city", "reference"]).size().reset_index()
        cityref_df.columns = ["city", "impression", "rcnt"]
        cityref_df = pd.merge(cityref_df, city_df, on="city", how="left")
        cityref_df["ctrbycity"] = cityref_df["rcnt"].astype(float) / cityref_df["ccnt"].astype(float)
        cityref_df = cityref_df[["city", "impression", "ctrbycity"]]
        X = pd.merge(X, cityref_df, on=["city", "impression"], how="left")
        X["ctrbycity"] = X["ctrbycity"].fillna(0)
        del cr_tmp_df
        del city_df
        del cityref_df

        # item ctr by city rank
        ctrbycity_rank_df = X[["session_id", "ctrbycity"]].copy()
        ctrbycity_rank_df = ctrbycity_rank_df[["session_id", "ctrbycity"]].groupby("session_id").rank(ascending=False)
        ctrbycity_rank_df.columns = ["ctrbycity_rank"]
        X = pd.concat([X, ctrbycity_rank_df], axis=1)
        del ctrbycity_rank_df

        # bayes likelihood by item
        bayes_likelihood = dataset["bayes_likelihood"]
        X["rlr"] = X["impression"].astype(str) + X["last_reference"].astype(str)
        def set_bayes_li(rlr):
            if rlr in bayes_likelihood:
                return bayes_likelihood[rlr]
            return 0.0
        X["bayes_li"] = X[["rlr"]].apply(lambda x: set_bayes_li(x.rlr), axis=1)

        # some action_types are already done by each item during each session
        couted_df = all_df[["action_type", "session_id", "reference"]].copy()
        action_types = ["interaction item rating"
            , "interaction item info"
            , "interaction item image"
            , "interaction item deals"
            , "search for item"]
        ated_cols = ["iired"
            , "iifed"
            , "iiied"
            , "iided"
            , "sfied"]
        for i, action_type in enumerate(action_types):
            at_df = couted_df[couted_df.action_type == action_type].copy()
            at_df = at_df[["session_id", "reference"]]
            at_df.columns = ["session_id", "impression"]
            at_df = at_df[~at_df.duplicated()]
            at_df[ated_cols[i]] = 1
            X = pd.merge(X, at_df, on=["session_id", "impression"], how="left")
            X[ated_cols[i]] = X[ated_cols[i]].fillna(0)
            X[ated_cols[i]] = X[ated_cols[i]].astype(int)
            del at_df

        del couted_df

        # dropout rate by each item during each session
        dropout_df = all_df[["session_id", "action_type", "reference", "is_y"]].copy()
        dropout_df = dropout_df[dropout_df.action_type.isin(["interaction item image", "clickout item"])]
        dropout_df = dropout_df[dropout_df.is_y == 0] # to prevent leakage
        dropout_df.loc[dropout_df["action_type"] == "interaction item image", "iii"] = 1
        dropout_df["iii"] = dropout_df["iii"].fillna(0)
        dropout_df.loc[dropout_df["action_type"] == "clickout item", "cko"] = 1
        dropout_df["cko"] = dropout_df["cko"].fillna(0)
        def is_dropout(iii, cko):
            if iii != 0 and cko != 0:
                return 0
            elif iii != 0 and cko == 0:
                return 1
            else:
                return -1
        dropout_df = dropout_df[["session_id", "reference", "iii", "cko"]].groupby(["session_id", "reference"]).apply(
            lambda x: is_dropout(np.sum(x.iii), np.sum(x.cko))).reset_index()
        dropout_df.columns = ["session_id", "reference", "dropout"]
        dropout_df = dropout_df[dropout_df != -1]
        dropout_df = dropout_df[["reference", "dropout"]].groupby("reference").apply(
            lambda x: np.sum(x.dropout).astype(float) / len(x.dropout)).reset_index()
        dropout_df.columns = ["impression", "dropout_rate"]
        X = pd.merge(X, dropout_df, on="impression", how="left")
        del dropout_df

        # dropout rate by each item during all sessions
        action_types = ["interaction item rating"
            , "interaction item info"
            , "interaction item image"
            , "interaction item deals"
            , "search for item"]
        dropout_df = all_df[["session_id", "action_type", "reference", "is_y"]].copy()
        dropout_df = dropout_df[dropout_df.action_type.isin(action_types + ["clickout item"])]
        dropout_df = dropout_df[dropout_df.is_y == 0] # to prevent leakage
        dropout_df.loc[dropout_df.action_type.isin(action_types), "iii"] = 1
        dropout_df["iii"] = dropout_df["iii"].fillna(0)
        dropout_df.loc[dropout_df["action_type"] == "clickout item", "cko"] = 1
        dropout_df["cko"] = dropout_df["cko"].fillna(0)
        dropout_df = dropout_df[["session_id", "reference", "iii", "cko"]].groupby(["session_id", "reference"]).apply(
            lambda x: is_dropout(np.sum(x.iii), np.sum(x.cko))).reset_index()
        dropout_df.columns = ["session_id", "reference", "dropout"]
        dropout_df = dropout_df[dropout_df != -1]
        dropout_df = dropout_df[["reference", "dropout"]].groupby("reference").apply(
            lambda x: np.sum(x.dropout).astype(float) / len(x.dropout)).reset_index()
        dropout_df.columns = ["impression", "all_dropout_rate"]
        X = pd.merge(X, dropout_df, on="impression", how="left")
        del dropout_df

        # action_type rate by each item
        action_types = ["interaction item rating"
            , "interaction item info"
            , "interaction item image"
            , "interaction item deals"
            , "search for item"
            , "clickout item"]
        atstats_df = all_df[["session_id", "action_type", "reference", "is_y"]].copy()
        atstats_df = atstats_df[atstats_df.action_type.isin(action_types)]
        atstats_df = atstats_df[atstats_df.is_y == 0] # to prevent leakage
        atstats_df = atstats_df[["reference", "action_type"]].groupby(["reference", "action_type"]).size().reset_index()
        atstats_df.columns = ["reference", "action_type", "at_cnt"]
        atstats_refcnt_df = atstats_df[["reference", "at_cnt"]].groupby("reference").sum().reset_index()
        atstats_refcnt_df.columns = ["reference", "rf_cnt"]
        atstats_df = pd.merge(atstats_df, atstats_refcnt_df, on="reference", how="left")
        atstats_df["at_rate"] = atstats_df["at_cnt"].astype(float) / atstats_df["rf_cnt"]
        atstats_df = atstats_df.pivot(index='reference', columns='action_type', values='at_rate').reset_index()
        at_rate_cols = ["co_at_rate", "iid_at_rate", "iii_at_rate", "iif_at_rate", "iir_at_rate", "sfi_at_rate"]
        atstats_df.columns = ["impression"] + at_rate_cols
        atstats_df = atstats_df.fillna(0)
        X = pd.merge(X, atstats_df, on="impression", how="left")
        for at_rate_col in at_rate_cols:
            X[at_rate_col] = X[at_rate_col].fillna(0)
        del atstats_df

        # action_type rate in-session rank by each item
        at_rate_cols = ["co_at_rate"
            , "iid_at_rate"
            , "iii_at_rate"
            , "iif_at_rate"
            , "iir_at_rate"
            , "sfi_at_rate"]
        at_rank_cols = []
        for at_rate_col in at_rate_cols:
            at_rank_col = at_rate_col + "_rank"
            at_rank_cols.append(at_rank_col)
            at_rank_df = X[["session_id", at_rate_col]].copy()
            at_rank_df = at_rank_df[["session_id", at_rate_col]].groupby("session_id").rank(ascending=False)
            at_rank_df.columns = [at_rank_col]
            X = pd.concat([X, at_rank_df], axis=1)
            del at_rank_df

        # reference_elapsed_mean and by action_type
        action_types = ["interaction item rating"
            , "interaction item info"
            , "interaction item image"
            , "interaction item deals"
            , "search for item"
            , "clickout item"]
        relapsed_df = all_df[
            ["session_id", "step", "timestamp", "timestamp_dt", "action_type", "reference", "is_y"]].copy()
        relapsed_df["pre_timestamp"] = relapsed_df["timestamp"].shift(1)
        relapsed_df["pre_timestamp_dt"] = relapsed_df["timestamp_dt"].shift(1)
        relapsed_df["pre_session_id"] = relapsed_df["session_id"].shift(1)
        relapsed_df = relapsed_df[relapsed_df.session_id == relapsed_df.pre_session_id]
        relapsed_df["elapsed"] = relapsed_df["timestamp"] - relapsed_df["pre_timestamp"]
        relapsed_df = relapsed_df[relapsed_df.action_type.isin(action_types)]
        relapsed_df = relapsed_df[relapsed_df.is_y == 0] # to prevent leakage
        relapsed_df = relapsed_df[relapsed_df.elapsed.notna()]
        relapsed_df = relapsed_df[relapsed_df.elapsed > 0]
        r_relapsed_df = relapsed_df[["reference", "elapsed"]].groupby("reference").agg(
            {"elapsed": np.mean}).reset_index()
        r_relapsed_rate_cols = ["ref_elapsed_mean"]
        r_relapsed_df.columns = ["impression"] + r_relapsed_rate_cols
        a_relapsed_df = relapsed_df[["reference", "action_type", "elapsed"]].groupby(["reference", "action_type"]).agg(
            {"elapsed": np.mean}).reset_index()
        a_relapsed_df.columns = ["reference", "action_type", "at_elapsed_mean"]
        a_relapsed_df = a_relapsed_df.pivot(index='reference', columns='action_type',
                                            values='at_elapsed_mean').reset_index()
        a_relapsed_rate_cols = ["co_ref_elapsed_mean", "iid_ref_elapsed_mean", "iii_ref_elapsed_mean",
                                "iif_ref_elapsed_mean", "iir_ref_elapsed_mean", "sfi_ref_elapsed_mean"]
        a_relapsed_df.columns = ["impression"] + a_relapsed_rate_cols
        X = pd.merge(X, r_relapsed_df, on="impression", how="left")
        X = pd.merge(X, a_relapsed_df, on="impression", how="left")
        del relapsed_df
        del r_relapsed_df
        del a_relapsed_df

        # tsh "time split by hour" item ctr
        tsh_df = all_df[all_df.action_type == "clickout item"][
            ["session_id", "action_type", "reference", "timestamp_dt", "is_y"]].copy()
        tsh_df["tsh24"] = -1
        X["tsh24"] = -1
        ts_min = tsh_df["timestamp_dt"].min()
        ts_max = tsh_df["timestamp_dt"].max()
        def set_tscol(hours):
            tscol = "tsh" + str(hours)
            ts_start = ts_min
            ts_end = ts_start + datetime.timedelta(hours=hours)
            ts_bin = 1
            while True:
                tsh_df.loc[(tsh_df.timestamp_dt >= ts_start) & (tsh_df.timestamp_dt < ts_end), tscol] = ts_bin
                X.loc[(X.timestamp_dt >= ts_start) & (X.timestamp_dt < ts_end), tscol] = ts_bin
                ts_start = ts_end
                ts_end = ts_start + datetime.timedelta(hours=hours)
                if ts_start > ts_max:
                    break
                ts_bin += 1
        set_tscol(24)
        tsh_df = tsh_df[tsh_df.is_y == 0]
        tsh24_df = tsh_df[["tsh24"]].groupby(["tsh24"]).size().reset_index()
        tsh24_df.columns = ["tsh24", "allcnt"]
        tsh24ref_df = tsh_df[["tsh24", "reference"]].groupby(["tsh24", "reference"]).size().reset_index()
        tsh24ref_df.columns = ["tsh24", "impression", "rcnt"]
        tsh24ref_df = pd.merge(tsh24ref_df, tsh24_df, on="tsh24", how="left")
        tsh24ref_df["ctrbytsh24"] = tsh24ref_df["rcnt"].astype(float) / tsh24ref_df["allcnt"].astype(float)
        tsh24ref_df = tsh24ref_df[["tsh24", "impression", "ctrbytsh24"]]
        X = pd.merge(X, tsh24ref_df, on=["tsh24", "impression"], how="left")
        X["ctrbytsh24"] = X["ctrbytsh24"].fillna(0)
        del tsh_df
        del tsh24_df
        del tsh24ref_df

        # item ctr by some props
        ctrbyprops_df = all_df[all_df.action_type == "clickout item"][["session_id", "reference", "is_y"]].copy()
        ctrbyprops_df.columns = ["session_id", "item_id", "is_y"]
        star_cols = ["p1 Star", "p2 Star", "p3 Star", "p4 Star", "p5 Star"]
        rating_cols = ["pSatisfactory Rating", "pGood Rating", "pVery Good Rating", "pExcellent Rating"]
        ctrbyprops_df = pd.merge(ctrbyprops_df, item_props[["item_id"] + star_cols + rating_cols], on="item_id",
                                 how="left")
        ctrbyprops_df["star"] = -1
        ctrbyprops_df.loc[ctrbyprops_df["p1 Star"] == 1, "star"] = 1
        ctrbyprops_df.loc[ctrbyprops_df["p2 Star"] == 1, "star"] = 2
        ctrbyprops_df.loc[ctrbyprops_df["p3 Star"] == 1, "star"] = 3
        ctrbyprops_df.loc[ctrbyprops_df["p4 Star"] == 1, "star"] = 4
        ctrbyprops_df.loc[ctrbyprops_df["p5 Star"] == 1, "star"] = 5
        ctrbyprops_df["r6"] = 0
        ctrbyprops_df["r7"] = 0
        ctrbyprops_df["r8"] = 0
        ctrbyprops_df["r9"] = 0
        ctrbyprops_df.loc[ctrbyprops_df["pSatisfactory Rating"] == 1, "r6"] = 6
        ctrbyprops_df.loc[ctrbyprops_df["pGood Rating"] == 1, "r7"] = 7
        ctrbyprops_df.loc[ctrbyprops_df["pVery Good Rating"] == 1, "r8"] = 8
        ctrbyprops_df.loc[ctrbyprops_df["pExcellent Rating"] == 1, "r9"] = 9
        ctrbyprops_df["rating"] = ctrbyprops_df[["r6", "r7", "r8", "r9"]].apply(
            lambda x: np.mean(np.trim_zeros(np.array([x.r6, x.r7, x.r8, x.r9]))), axis=1)
        ctrbyprops_df["rating"] = ctrbyprops_df["rating"].fillna(-1)
        ctrbyprops_df["star_rating"] = "sr_" + ctrbyprops_df["star"].astype(str) + "_" + ctrbyprops_df["rating"].astype(
            str)
        ctrbyprops_df = ctrbyprops_df[["session_id", "star_rating", "item_id", "is_y"]]
        ctrbyprops_df = ctrbyprops_df[ctrbyprops_df.is_y == 0] # to prevent leakage
        ctrbyprops_df = ctrbyprops_df[["item_id", "star_rating"]]
        ctrbyprops_df.columns = ["impression", "star_rating"]
        prop_df = ctrbyprops_df[["star_rating"]].groupby(["star_rating"]).size().reset_index()
        prop_df.columns = ["star_rating", "allcnt"]
        propref_df = ctrbyprops_df[["star_rating", "impression"]].groupby(
            ["star_rating", "impression"]).size().reset_index()
        propref_df.columns = ["star_rating", "impression", "rcnt"]
        propref_df = pd.merge(propref_df, prop_df, on="star_rating", how="left")
        propref_df["ctrbyprops"] = propref_df["rcnt"].astype(float) / propref_df["allcnt"].astype(float)
        propref_df = propref_df[["star_rating", "impression", "ctrbyprops"]]
        X["star_rating"] = "sr_" + X["star"].astype(str) + "_" + X["rating"].astype(str)
        X = pd.merge(X, propref_df, on=["star_rating", "impression"], how="left")
        X["ctrbyprops"] = X["ctrbyprops"].fillna(0)
        del ctrbyprops_df
        del prop_df
        del propref_df

        # is no serach item
        action_types = ["clickout item"]
        is_nosi_df = all_df[["session_id", "action_type", "reference", "is_y"]].copy()
        is_nosi_df = is_nosi_df.groupby("session_id").first().reset_index()
        is_nosi_df = is_nosi_df[(is_nosi_df.action_type.isin(action_types)) & (is_nosi_df.is_y == 0)]
        is_nosi_df = is_nosi_df[["reference"]].groupby("reference").size().reset_index()
        is_nosi_df.columns = ["impression", "nosearch_cnt"]
        X = pd.merge(X, is_nosi_df, on="impression", how="left")
        X["nosearch_cnt"] = X["nosearch_cnt"].fillna(0)
        del is_nosi_df

        return X

class BySession(object):
    @classmethod
    def set(cls, X, dataset):
        print("... ... BySession")
        all_df = dataset["all_df"]

        # item ratio of appearance by each session
        def get_precnt_ratio(x):
            pre_references = str(x.pre_references).split("|")
            len_pre_ref = len(pre_references)
            if len_pre_ref != 0:
                return np.float(pre_references.count(x.impression)) / len_pre_ref
            return 0

        preref_df = all_df[all_df.action_type != "clickout item"].groupby("session_id").apply(
            lambda x: "|".join([r for r in list(x.reference) if str.isnumeric(r)])).reset_index()
        preref_df.columns = ["session_id", "pre_references"]
        X = pd.merge(X, preref_df, on="session_id", how="left")
        X[["pre_references"]] = X[["pre_references"]].fillna("")
        X["precnt_ratio"] = X[["impression", "pre_references"]].apply(lambda x: get_precnt_ratio(x), axis=1)
        del preref_df

        # action_type ratio of appearance by each session
        atype_long_names = ["interaction item rating"
            , "interaction item info"
            , "interaction item image"
            , "interaction item deals"
            , "search for item"
            , "clickout item"]
        atype_short_names = ["iir_ratio"
            , "iif_ratio"
            , "iii_ratio"
            , "iid_ratio"
            , "sfi_ratio"
            , "co_ratio"]
        preref_df2 = all_df[all_df.action_type.isin(atype_long_names)][
            ["session_id", "reference", "action_type", "is_y"]].copy()
        preref_df2 = preref_df2[preref_df2.is_y == 0] # to prevent leakage
        preref_df2 = preref_df2[["session_id", "reference", "action_type"]]
        preref_df3 = preref_df2[["session_id"]].groupby("session_id").size().reset_index()
        preref_df3.columns = ["session_id", "cnt"]
        preref_df2 = pd.get_dummies(preref_df2, columns=['action_type'])
        preref_df2 = preref_df2.groupby(["session_id", "reference"]).sum().reset_index()
        preref_df2.columns = ["session_id", "impression"] + atype_short_names
        preref_df2 = pd.merge(preref_df2, preref_df3, on="session_id", how="left")
        preref_df2[atype_short_names] = preref_df2[atype_short_names].astype(float)
        for atype_short_name in atype_short_names:
            preref_df2[atype_short_name] = preref_df2[atype_short_name] / preref_df2["cnt"]
        X = pd.merge(X, preref_df2, on=["session_id", "impression"], how="left")
        del preref_df2
        del preref_df3

        # clickouted item during session
        couted_df = all_df[["action_type", "session_id", "reference", "is_y"]].copy()
        couted_df = couted_df[couted_df.action_type == "clickout item"]
        couted_df = couted_df[couted_df.is_y == 0] # to prevent leakage
        couted_df = couted_df[["session_id", "reference"]]
        couted_df.columns = ["session_id", "impression"]
        couted_df = couted_df[~couted_df.duplicated()]
        couted_df["clickouted"] = 1
        X = pd.merge(X, couted_df, on=["session_id", "impression"], how="left")
        X["clickouted"] = X["clickouted"].fillna(0)
        X["clickouted"] = X["clickouted"].astype(int)

        # clickouted item 2 item during session
        v2v_counter = dataset["v2v_counter"]
        def extract_sv2v_counter(iids):
            v = {}
            for iid in iids:
                if iid in v2v_counter:
                    for s in v2v_counter[iid]:
                        if not s in v:
                            v[s] = v2v_counter[iid][s]
            return v

        sv2v_df = couted_df.groupby("session_id").apply(
            lambda x: extract_sv2v_counter(list(x.impression))).reset_index()
        sv2v_df.columns = ["session_id", "sv2v"]
        X = pd.merge(X, sv2v_df, on="session_id", how="left")
        X["sv2v"] = X["sv2v"].fillna("{}")
        X["sv2v_score"] = X[["impression", "sv2v"]].apply(
            lambda x: x.sv2v[x.impression] if x.impression in x.sv2v else np.nan, axis=1)
        X.drop("sv2v", axis=1, inplace=True)
        sv2vs_stats = X.groupby("session_id").agg({"sv2v_score": [np.mean, np.std]}).reset_index()
        sv2vs_stats.columns = ["session_id", "sv2v_score_mean", "sv2v_score_std"]
        X = pd.merge(X, sv2vs_stats, on="session_id", how="left")
        X["sv2v_score_norm"] = X["sv2v_score"] - X["sv2v_score_mean"] / X["sv2v_score_std"]
        del couted_df
        del sv2v_df
        del sv2vs_stats

        # is zero interactions
        zeroit_df = all_df[["session_id"]].groupby("session_id").size().reset_index()
        zeroit_df.columns = ["session_id", "it_count"]
        zeroit_df["is_zeroit"] = zeroit_df[["it_count"]].apply(lambda x: 1 if x.it_count == 1 else 0, axis=1)
        X = pd.merge(X, zeroit_df, on="session_id", how="left")
        del zeroit_df

        # diff between clickouted price mean
        co_price_df = all_df[all_df.action_type == "clickout item"][
            ["session_id", "reference", "prices", "impressions", "is_y"]].copy()
        co_price_df = co_price_df[co_price_df.is_y == 0] # to prevent leakage
        def get_price(reference, impressions, prices):
            imps = str(impressions).split("|")
            prs = str(prices).split("|")
            if reference in imps:
                return prs[imps.index(reference)]
            else:
                return 0
        co_price_df["price"] = co_price_df.apply(lambda x: get_price(x.reference, x.impressions, x.prices), axis=1)
        co_price_df["price"] = co_price_df["price"].astype(float)
        co_price_df = co_price_df.groupby("session_id").agg({'price': np.mean}).reset_index()
        co_price_df.columns = ["session_id", "couted_price_mean"]
        X = pd.merge(X, co_price_df, on="session_id", how="left")
        X["couted_price_mean"] = X["couted_price_mean"].fillna(-1)
        X["co_price_diff"] = X["price"].astype(float) / X["couted_price_mean"]
        X.loc[X.co_price_diff < 0, "co_price_diff"] = 0
        del co_price_df

        # first action_type
        firsta_df = all_df[["session_id", "_session_id", "action_type", "is_y"]].copy()
        firsta_df = firsta_df[firsta_df.is_y == 0] # to prevent leakage
        firsta_df = firsta_df.groupby("_session_id").first().reset_index()
        firsta_df = firsta_df.groupby("session_id").last().reset_index()
        firsta_df.loc[firsta_df["action_type"] == "search for destination", "action_type"] = "fa_sfd"
        firsta_df.loc[firsta_df["action_type"] == "interaction item image", "action_type"] = "fa_iii"
        firsta_df.loc[firsta_df["action_type"] == "clickout item", "action_type"] = "fa_coi"
        firsta_df.loc[firsta_df["action_type"] == "search for item", "action_type"] = "fa_sfi"
        firsta_df.loc[firsta_df["action_type"] == "search for poi", "action_type"] = "fa_sfp"
        firsta_df.loc[firsta_df["action_type"] == "change of sort order", "action_type"] = "fa_coso"
        firsta_df.loc[firsta_df["action_type"] == "filter selection", "action_type"] = "fa_fis"
        firsta_df.loc[firsta_df["action_type"] == "interaction item info", "action_type"] = "fa_iiinfo"
        firsta_df.loc[firsta_df["action_type"] == "interaction item rating", "action_type"] = "fa_iirat"
        firsta_df.loc[firsta_df["action_type"] == "interaction item deals", "action_type"] = "fa_iidea"
        firsta_df = firsta_df[["session_id", "action_type"]]
        firsta_df.columns = ["session_id", "at"]
        onehot_firsta = pd.get_dummies(firsta_df, columns=['at'])
        firsta_cols = list(onehot_firsta.columns)
        firsta_cols.remove("session_id")
        X = pd.merge(X, onehot_firsta, on="session_id", how="left")
        for firsta_col in firsta_cols:
            X[firsta_col] = X[firsta_col].fillna(0)
        del firsta_df
        del onehot_firsta

        # unique reference ratio during session
        uniqueref_df = all_df[["session_id", "reference", "action_type", "is_y"]].copy()
        action_types = ["interaction item rating"
            , "interaction item info"
            , "interaction item image"
            , "interaction item deals"
            , "search for item"
            , "clickout item"]
        uniqueref_df = uniqueref_df[uniqueref_df.action_type.isin(action_types)]
        uniqueref_df = uniqueref_df[uniqueref_df.is_y == 0] # to prevent leakage
        uniqueref_df = uniqueref_df[["session_id", "reference"]].groupby("session_id").apply(
            lambda x: len(set(list(x.reference))) / len(list(x.reference))).reset_index()
        uniqueref_df.columns = ["session_id", "uniqueref_ratio"]
        X = pd.merge(X, uniqueref_df, on="session_id", how="left")
        del uniqueref_df

        # number of action_types during session
        action_types = ["interaction item rating"
            , "interaction item info"
            , "interaction item image"
            , "interaction item deals"
            , "search for item"
            , "clickout item"]
        atcnt_cols = ["iir_cnt"
            , "iif_cnt"
            , "iii_cnt"
            , "iid_cnt"
            , "sfi_cnt"
            , "co_cnt"]
        cocnt_df = all_df[all_df.action_type.isin(action_types)][["session_id", "action_type", "is_y"]].copy()
        cocnt_df = cocnt_df[cocnt_df.is_y == 0] # to prevent leakage
        for i, action_type in enumerate(action_types):
            cnt_df = cocnt_df[cocnt_df.action_type == action_type][["session_id"]].copy()
            cnt_df = cnt_df[["session_id"]].groupby("session_id").size().reset_index()
            cnt_df.columns = ["session_id", atcnt_cols[i]]
            X = pd.merge(X, cnt_df, on="session_id", how="left")
            X[atcnt_cols[i]] = X[atcnt_cols[i]].fillna(0)
            X[atcnt_cols[i]] = X[atcnt_cols[i]].astype(int)
            del cnt_df
        del cocnt_df

        # last duration all "is it same as is_last_elapsed_time?"
        lduration_all_df = all_df[["session_id", "action_type", "timestamp", "is_y"]].copy()
        lduration_all_df["pre_timestamp"] = lduration_all_df["timestamp"].shift(1)
        lduration_all_df["pre_session_id"] = lduration_all_df["session_id"].shift(1)
        lduration_all_df = lduration_all_df[lduration_all_df.session_id == lduration_all_df.pre_session_id]
        lduration_all_df["elapsed_for_all"] = lduration_all_df["timestamp"] - lduration_all_df["pre_timestamp"]
        lduration_all_df = lduration_all_df[lduration_all_df.is_y == 1]
        lduration_all_df = lduration_all_df[["session_id", "elapsed_for_all"]]
        X = pd.merge(X, lduration_all_df, on="session_id", how="left")
        del lduration_all_df

        # click out cnt by all and to summarize by session")
        cocntbyss_df = all_df[(all_df.action_type == "clickout item") & (all_df.is_y == 0)][["reference"]].copy()
        cocnt_df = cocntbyss_df.groupby(["reference"]).size().reset_index()
        cocnt_df.columns = ["impression", "cocntall"]
        X = pd.merge(X, cocnt_df, on="impression", how="left")
        X["cocntall"] = X["cocntall"].fillna(0)
        cocntbyday_stats_df = X[["session_id", "cocntall"]].groupby("session_id").agg(
            {'cocntall': [np.mean, np.std]}).reset_index()
        cocntbyday_stats_df.columns = ["session_id", "cocntall_mean", "cocntall_std"]
        X = pd.merge(X, cocntbyday_stats_df, on="session_id", how="left")
        X["cocntall_norm"] = (X["cocntall"] - X["cocntall_mean"]) / X["cocntall_std"]
        del cocntbyss_df
        del cocnt_df
        del cocntbyday_stats_df

        return X


class EncodingForCategories(object):
    @classmethod
    def to_prob(cls, X, dataset):
        print("... ... EncodingForCategories")
        all_df = dataset["all_df"]

        # city prob
        city_df = all_df[["city"]].copy()
        city_vc_df = city_df["city"].value_counts().reset_index()
        city_vc_df.columns = ["city", "cnt"]
        city_vc_df["city_prob"] = city_vc_df["cnt"].astype(float) / city_df.shape[0]
        city_vc_df = city_vc_df[["city", "city_prob"]]
        X = pd.merge(X, city_vc_df, on="city", how="left")
        del city_df
        del city_vc_df

        # platform prob
        plt_df = all_df[["platform"]].copy()
        plt_vc_df = plt_df["platform"].value_counts().reset_index()
        plt_vc_df.columns = ["platform", "cnt"]
        plt_vc_df["platform_prob"] = plt_vc_df["cnt"].astype(float) / plt_df.shape[0]
        plt_vc_df = plt_vc_df[["platform", "platform_prob"]]
        X = pd.merge(X, plt_vc_df, on="platform", how="left")
        del plt_df
        del plt_vc_df

        return X


class ByLocation(object):

    @classmethod
    def set(cls, X, dataset):
        print("... ... ByLocation")
        # join city country prob
        all_df = dataset["all_df"]
        city_df = all_df[["city", "country_name"]].copy()
        city_vc_df = city_df["country_name"].value_counts().reset_index()
        city_vc_df.columns = ["country_name", "cnt"]
        city_vc_df["country_name_prob"] = city_vc_df["cnt"].astype(float) / city_df.shape[0]
        city_vc_df = city_vc_df[["country_name", "country_name_prob"]]
        X = pd.merge(X, city_vc_df, on="country_name", how="left")
        del city_df
        del city_vc_df

        # join ctr by platform
        plt_tmp_df = all_df[all_df.action_type == "clickout item"].copy()
        plt_tmp_df = plt_tmp_df[plt_tmp_df.is_y == 0] # to prevent leakage
        plt_df = plt_tmp_df[["platform"]].groupby(["platform"]).size().reset_index()
        plt_df.columns = ["platform", "pcnt"]
        pltref_df = plt_tmp_df[["platform", "reference"]].groupby(["platform", "reference"]).size().reset_index()
        pltref_df.columns = ["platform", "impression", "rcnt"]
        pltref_df = pd.merge(pltref_df, plt_df, on="platform", how="left")
        pltref_df["ctrbyplatform"] = pltref_df["rcnt"].astype(float) / pltref_df["pcnt"].astype(float)
        pltref_df = pltref_df[["platform", "impression", "ctrbyplatform"]]
        X = pd.merge(X, pltref_df, on=["platform", "impression"], how="left")
        X["ctrbyplatform"] = X["ctrbyplatform"].fillna(0)
        del plt_tmp_df
        del plt_df
        del pltref_df

        # join ctr by platform rank
        ctrbyp_rank_df = X[["session_id", "ctrbyplatform"]].copy()
        ctrbyp_rank_df = ctrbyp_rank_df[["session_id", "ctrbyplatform"]].groupby("session_id").rank(ascending=False)
        ctrbyp_rank_df.columns = ["ctrbyplatform_rank"]
        X = pd.concat([X, ctrbyp_rank_df], axis=1)
        del ctrbyp_rank_df

        return X


class Polinomials(object):
    @classmethod
    def set(cls, X):
        print("... ... Polinomials")
        X["ild_x_pr"] = X["is_last_duration"] * X["pos_rate"]
        return X

class TargetVariable(object):
    @classmethod
    def set(cls, X):
        print("... ... TargetVariable")
        X["clicked"] = X[["impression", "reference"]].apply(lambda x: 1 if x.impression == x.reference else 0, axis=1)
        return X
