'''
Returns dataset after L1-based feature selection

EXAMPLES:

Classification of text documents using sparse features:
Comparison of different algorithms for document classification
including L1-based feature selection.
'''
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

def L1_based(X, Y):
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X)
    print('L1-Based Feature Selection: ', X_new.shape)
    return X_new
    


if __name__ == "__main__":
    main()
