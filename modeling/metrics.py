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