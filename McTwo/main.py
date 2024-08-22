import numpy as np
import pandas as pd
import os
from McOne import McOne
from McTwo import McTwo
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


def evaluation(X, y):
    y = y.astype('int')
    _5fold = KFold(n_splits=5)
    mAcc = []
    for train_index, test_index in _5fold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        acc1 = np.mean(SVC().fit(X_train, y_train).predict(X_test) == y_test)
        acc2 = np.mean(GaussianNB().fit(X_train, y_train).predict(X_test) == y_test)
        acc3 = np.mean(DecisionTreeClassifier().fit(X_train, y_train).predict(X_test) == y_test)
        acc4 = np.mean(KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train).predict(X_test) == y_test)
        mAcc.append(max(acc1, acc2, acc3, acc4))
    return np.array(mAcc).mean()


dataPath = "../data/train"
classPath = "../data/class"
file = list(zip(sorted(os.listdir(dataPath)), sorted(os.listdir(classPath))))
res = { "DataSet": [],
        "McOne_Acc": [],
        "McTwo_Acc": []
    }
for dataName, className in file:
    # print(f'DataSet: {dataName} ClassSet: {className}
    data = pd.read_csv(os.path.join(dataPath, dataName)).transpose().values
    data_class = pd.read_csv(os.path.join(classPath, className)).values
    features = data[1:, :]
    label = data_class[:, 1]
    for idx, l in enumerate(list(set(label))):
        label[np.where(label == l)] = idx
    FOne = McOne(features, label, 0.2)
    FTwo = McTwo(FOne, label)
    mAcc1 = evaluation(FOne, label)
    mAcc2 = evaluation(FTwo, label)
    res["DataSet"].append(dataName)
    res["McOne_Acc"].append(mAcc1)
    res["McTwo_Acc"].append(mAcc2)
    # print(f'MCOne.Acc: {mAcc1}, MCTwo.Acc: {mAcc2}')

pd.DataFrame(res).to_csv("McTwoRes.csv")



