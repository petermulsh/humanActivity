from trainingData import trainingDataWithoutLabel, trainingDataActivity, testDataWithoutLabel, testDataActivity
from svd import svd

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
    '''SVD based feature selection'''
    New_features = trainingDataWithoutLabel()
    test_features= testDataWithoutLabel()
    
    print('SVD', New_features.shape)
    testKNNAlgorithm(New_features, trainingDataActivity())


def testKNNAlgorithm(X, Y):
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=7)
    cv_results = model_selection.cross_val_score(KNeighborsClassifier(), X, Y, cv=kfold, scoring='accuracy')
    msg = "%s: %f (%f)" % ('KNN', cv_results.mean(), cv_results.std())
    print(msg)

if __name__ == "__main__":
    main()
