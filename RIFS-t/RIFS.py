import random
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import pandas as pd
import os
import numpy as np


def sIFS(feature_ranks, classifier, X, y, k=0, D=1):
    best_feature_subset = []
    current_subset = []
    decrease_times = 0
    max_accuracy = 0
    y = y.astype('int')
    while decrease_times <= D and k < len(feature_ranks):
        next_feature = feature_ranks[k]
        current_subset.append(next_feature)
        accuracy = cross_val_score(classifier, X[:, current_subset], y, cv=10).mean()
        if accuracy > max_accuracy:
            best_feature_subset = current_subset[:]
            max_accuracy = accuracy
            decrease_times = 0
        else:
            decrease_times += 1
        k += 1
    return best_feature_subset, max_accuracy


def RIFS(feature_ranks, classifier, X, y, K=10, D=1):
    best_overall_subset = []
    max_accuracy = 0
    for _ in range(K):
        k = random.randint(0, len(feature_ranks) - 1)
        subset,accuracy = sIFS(feature_ranks, classifier, X, y, k, D)
        if accuracy > max_accuracy:
            best_overall_subset = subset[:]
            max_accuracy = accuracy
    return best_overall_subset, max_accuracy






