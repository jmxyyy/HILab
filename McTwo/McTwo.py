import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def KNN_BAcc(F, L):
    s, _ = F.shape
    L = L.astype('int')
    NN = KNeighborsClassifier(n_neighbors=1)
    res = []
    for i in range(s):
        NN.fit(F[[x for x in range(s) if x != i]],
               L[[x for x in range(s) if x != i]])
        res.append(NN.predict(F[[i]]).tolist()[0])
    res = np.array(res)
    BAcc = (np.mean(res[np.where(L == 0)] == L[np.where(L == 0)]) +
            np.mean(res[np.where(L == 1)] == L[np.where(L == 1)])) / 2
    return BAcc


def McTwo(F, L):
    s, k = F.shape
    curBAcc = -1
    curSet = set([])
    leftSet = set([x for x in range(k)])
    while True:
        tempBAcc, idx = -1, -1
        for x in leftSet:
            BAcc = KNN_BAcc(F[:, list(curSet) + [x]], L)
            if BAcc > tempBAcc:
                tempBAcc = BAcc
                idx = x
        if tempBAcc > curBAcc:
            curBAcc = tempBAcc
            curSet.add(idx)
            leftSet.remove(idx)
        else:
            break
    return F[:, list(curSet)]