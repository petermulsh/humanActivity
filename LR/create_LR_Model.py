from trainingData import trainingDataWithoutLabel, trainingDataActivity, testDataWithoutLabel, testDataActivity

from sklearn.metrics import accuracy_score
import timeit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

import matplotlib.pyplot as plt
import numpy as np


'''Algorithm Test'''
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def main():
    New_features = trainingDataWithoutLabel()
    test_features= testDataWithoutLabel()
    testLRAlgorithm(New_features, trainingDataActivity())


def testLRAlgorithm(X, Y):
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=7)
    cv_results = model_selection.cross_val_score(LogisticRegression(), X, Y, cv=kfold, scoring='accuracy')
    msg = "%s: %f (%f)" % ('LR', cv_results.mean(), cv_results.std())
    print(msg)

if __name__ == "__main__":
    main()
