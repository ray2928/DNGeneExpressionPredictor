import sys
import numpy
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":
    data_true = numpy.loadtxt(sys.argv[1])
    print "data true", data_true.shape
    data_est = numpy.loadtxt(sys.argv[2])
    print "data est", data_est.shape
    residual = numpy.mean(data_true-data_est, axis = 1)
    print "residual", residual.shape
    numpy.savetxt(sys.argv[3], residual, fmt="%f")
