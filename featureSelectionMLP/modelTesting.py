from trainingData import trainingDataWithoutLabel, trainingDataActivity, testDataWithoutLabel, testDataActivity
from tree_based_featureSelection import tree_based
from L1_based_featureSelection import L1_based
from svd import svd

from sklearn.metrics import accuracy_score
import timeit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

import matplotlib.pyplot as plt
import numpy as np

'''L1 Based'''
from sklearn.svm import LinearSVC
'''Tree Based'''
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


'''Algorithm Test'''
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def main():
    testSVD()
    testL1()
    testTree()
    

def testSVD():
    '''SVD based feature selection'''
    features = trainingDataWithoutLabel()
    New_features = svd(features)
    test_features= svd(testDataWithoutLabel())
    
    print('SVD', New_features.shape)
    testModel(New_features, test_features)

def testL1():
    features = trainingDataWithoutLabel()
    label = trainingDataActivity()
    
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(features, label)
    model = SelectFromModel(lsvc, prefit=True)
    New_features = model.transform(features)
    test_features = model.transform(testDataWithoutLabel())
    
    print('L1',New_features.shape)
    testModel(New_features, test_features)

def testTree():
    features = trainingDataWithoutLabel()
    label = trainingDataActivity()

    clf = ExtraTreesClassifier()
    clf = clf.fit(features, label)
    model = SelectFromModel(clf, prefit=True)
    New_features = model.transform(features)
    test_features= model.transform(testDataWithoutLabel())
    
    print('Tree',New_features.shape)
    testModel(New_features, test_features)
    
def testModel(New_features, test_features):
    test = testDataActivity()
    label = trainingDataActivity()
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50, 20), random_state=1, shuffle=True, max_iter=400)
    start_time = timeit.default_timer()
    fit=clf.fit(New_features,label)
    pred=fit.predict(test_features)
    elapsed = timeit.default_timer() - start_time

    print(classification_report(test, pred, target_names=None, sample_weight=None, digits=4))
    print('Elapsed time: ', elapsed)


if __name__ == "__main__":
    main()
