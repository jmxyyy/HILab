import pandas as pd
from minepy import MINE
import numpy as np


def MIC(x, y):
    mine = MINE()
    mine.compute_score(x, y)
    return mine.mic()


def McOne(F, L, r = 0.2):
    s, k = F.shape
    micFC = [-1 for _ in range(k)]
    Subset = [-1 for _ in range(k)]
    numSubset = 0
    for i in range(k):
        micFC[i] = MIC(F[:, i], L)
        if micFC[i] >= r:
            Subset[numSubset] = i
            numSubset += 1
    Subset = Subset[0:numSubset]
    Subset.sort(key=lambda x: micFC[x], reverse=True)
    mask = [True for _ in range(numSubset)]
    for e in range(numSubset):
        if mask[e]:
            for q in range(e + 1, numSubset):
                if mask[q] and MIC(F[:, Subset[e]], F[:, Subset[q]]) >= micFC[Subset[q]]:
                    mask[q] = False
    FReduce = F[:, np.array(Subset)[mask]]
    return FReduce