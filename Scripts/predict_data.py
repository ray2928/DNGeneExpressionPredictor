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
    x = np.loadtxt(input).T 
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    y = np.loadtxt(output).T
	x = x[start:stop, :]
    y = y[start:stop, :]
    print "Size of input data: ", x.shape
    print "Size of target data: ", y.shape
	cv = cross_validation.KFold(len(x), folds = 5, indices=False)
	xtraining = []
	xvalidating = []
	ytraining = []
	yvalidating = []
	for traincv, testcv in cv:
		xtraining = x[traincv]
		print "Size of data: ", xtraining.shape
		xvalidating = x[testcv]
		print "Size of data: ", xvalidating.shape
		ytraining = y[traincv]
		print "Size of data: ", ytraining.shape
		yvalidating = y[testcv]
		print "Size of data: ", yvalidating.shape
		break
	x = np.concatenate((xtraining, xvalidating), axis=0)
	y = np.concatenate((ytraining, yvalidating), axis=0)
	ytraining = y[traincv]
	print "Size of final data: ", x.shape
	yvalidating = y[testcv]
	print "Size of final data: ", y.shape
    print "===========Finish Loading Data=============="
	raw_input()
    return DenseDesignMatrix(X=x, y=y)
