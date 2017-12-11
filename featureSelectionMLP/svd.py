'''
Returns dataset after SVD feature selection

EXAMPLES:

Feature importances with forests of trees:
example on synthetic data showing the recovery
of the actually meaningful features.

Pixel importances with a parallel forest of trees:
example on face recognition data.
'''
import numpy as np
import matplotlib.pyplot as plt
import pylab as plt

import sklearn
from sklearn import decomposition


def svd(X):
    U, s, V = np.linalg.svd(X.as_matrix(), full_matrices=False)
    #return U[:,0:5]
    '''SVD With KNN'''
    ##10 = 0.929000
    ##50 = 0.951442
    ##100 = 0.951577
    return U[:,0:50]

