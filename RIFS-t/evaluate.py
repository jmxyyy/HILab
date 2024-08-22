import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


def evaluate(X, y):
    y = y.astype('int')
    mAcc = []
    _10fold = KFold(n_splits=10)
    for train_index, test_index in _10fold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        acc1 = np.mean(SVC().fit(X_train, y_train).predict(X_test) == y_test)
        acc2 = np.mean(GaussianNB().fit(X_train, y_train).predict(X_test) == y_test)
        acc3 = np.mean(DecisionTreeClassifier().fit(X_train, y_train).predict(X_test) == y_test)
        acc4 = np.mean(KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train).predict(X_test) == y_test)
        acc5 = np.mean(LogisticRegression().fit(X_train, y_train).predict(X_test) == y_test)
        mAcc.append(max(acc1, acc2, acc3, acc4, acc5))
    return np.array(mAcc).mean()
