# We'll need numpy to manage arrays of data
import numpy as np
 
# We'll need the DenseDesignMatrix class to return the data
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
import time

def load_data(start, stop, input, output):
    print "===========Start Loading Data=============="
    x = np.loadtxt(input)
    print "Size of input data: ", x.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    y = np.loadtxt(output)
    print "Size of label data: ", y.shape
    print "===========Finish Loading Data=============="
    x = x[start:stop, :]
    y = y[start:stop, :]
 
    return DenseDesignMatrix(X=x, y=y)
