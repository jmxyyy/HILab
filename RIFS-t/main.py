import numpy as np
import pandas as pd
import os
import random
from evaluate import evaluate
from RIFS import RIFS
from sklearn.svm import SVC


dataPath = "../data/train"
classPath = "../data/class"
file = list(zip(sorted(os.listdir(dataPath)), sorted(os.listdir(classPath))))
res = { "DataSet": [],
        "Others_Acc": [],
        "RIFS_Acc": []
    }
for dataName, className in file:
    print(f"DataSet: {dataName}")
    data = pd.read_csv(os.path.join(dataPath, dataName)).transpose().values
    data_class = pd.read_csv(os.path.join(classPath, className)).values
    features = data[1:, :]
    label = data_class[:, 1]
    for idx, l in enumerate(list(set(label))):
        label[np.where(label == l)] = idx
    mAcc1 = evaluate(features, label)
    feature_ranks = list(range(features.shape[1]))
    random.shuffle(feature_ranks)
    best_subset,mAcc2 = RIFS(feature_ranks, SVC(), features, label)
    # print(f"{mAcc1}")
    # print(f"最佳特征子集: {best_subset}, 准确率: {mAcc2}")
    res["DataSet"].append(dataName)
    res["Others_Acc"].append(mAcc1)
    res["RIFS_Acc"].append(mAcc2)

pd.DataFrame(res).to_csv("RIFS_Res.csv")
