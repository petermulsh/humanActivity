'''
Returns dataset after Tree-based feature selection

EXAMPLES:

Feature importances with forests of trees:
example on synthetic data showing the recovery
of the actually meaningful features.

Pixel importances with a parallel forest of trees:
example on face recognition data.
'''
import pandas
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

def tree_based(X, Y):
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, Y) 
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print('Tree Based Feature Selection: ', X_new.shape)
    return X_new
    


if __name__ == "__main__":
    main()
