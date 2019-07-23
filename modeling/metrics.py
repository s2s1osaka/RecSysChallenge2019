import pandas as pd
import numpy as np
import datetime

def calc_mrr(x):
    impressions = list(x.impression)
    probs = list(x.prob)
    clickeds = list(x.clicked)
    if 1 not in clickeds:
        return 1.0
    reference = impressions[clickeds.index(1)]
    imps_probs = [[i, p] for i, p in zip(impressions, probs)]
    imps_probs.sort(key=lambda z: -z[1])
    return (1./([i[0] for i in imps_probs].index(reference)+1))

def nDCG(y_true, y_pred, is_1d=True, k=5):
    FIX_NCG_LOGS = {}
    for n in range(5):
        FIX_NCG_LOGS[n+1] = np.array([math.log(i+2, 2) for i in range(n+1)])
    scores = {}
    y_pred = np.array(y_pred)
    ndcgs = []
    def get_relevance(y, yp):
        rel = []
        sorted_yp = sorted(yp, reverse=True)
        for sorted_y in sorted_yp[:k]:
            rel.append(np.where(yp==sorted_y)[0][0])
        rel = np.array(rel)
        rel[rel==y] = -1
        rel[rel>-1] = 0
        rel[rel==-1] = 1
        return rel

    def calc_nDCG(rel):
        ideal_rel = np.array(sorted(rel, reverse=True))
        return sum(rel/FIX_NCG_LOGS[rel.shape[0]])/sum(ideal_rel/FIX_NCG_LOGS[rel.shape[0]])

    if is_1d:
        for y, yp in zip(np.array(y_true), y_pred):
            ndcgs.append(calc_nDCG(get_relevance(y, yp)))
    else:
        for y, yp in zip(np.where(y_true==1)[1], y_pred):
            ndcgs.append(calc_nDCG(get_relevance(y, yp)))

    ndcgs = np.nan_to_num(np.array(ndcgs))
    scores["nDCG@{}".format(k)] = np.mean(ndcgs)
    return scores

def calc_ndcg(df):
    aaa = df.copy()
    bbb = aaa[["gid", "prob"]].groupby("gid").apply(lambda x: rankdata(list(x.prob)) / np.sum([n+1 for n in range(len(list(x.prob)))])).reset_index()
    bbb.columns = ["gid", "predictions"]
    ccc = aaa[["gid", "clicked"]].groupby("gid").apply(lambda x: list(x.clicked).index(1) if 1 in list(x.clicked) else -1).reset_index()
    ccc.columns = ["gid", "ground_truth"]
    ddd = pd.merge(bbb, ccc)
    ndcg_arr = []
    ndcg_arr.append(nDCG(ddd.ground_truth.values, ddd.predictions.values, k=1))
    ndcg_arr.append(nDCG(ddd.ground_truth.values, ddd.predictions.values, k=3))
    ndcg_arr.append(nDCG(ddd.ground_truth.values, ddd.predictions.values, k=5))
    return ndcg_arr