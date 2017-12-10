'''
Returns the training data
'''
import pandas


def trainingDataWithoutLabel():
    data = pandas.read_csv("train.csv")
    print('Train Data Shape: ', data.shape)
    return data.iloc[:, :562]

def trainingDataActivity():
    data = pandas.read_csv("train.csv")
    return data['Activity']

def testDataWithoutLabel():
    data = pandas.read_csv("test.csv")
    print('Test Data Shape: ', data.shape)
    return data.iloc[:, :562]

def testDataActivity():
    data = pandas.read_csv("test.csv")
    return data['Activity']
