import sys
import numpy
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
def readInputfile(filename):
    print "hello"
    f = open(filename, 'r')
    i = 0
    x = []
    #store the data into list
    for line in f:
	print "???"
        if i >= 1:
            arr = []
            list = line.split()
	    print len(list)
            for j in range(1, len(list)):
                arr.append(int(list[j]))
            x.append(arr)
        i += 1
    f.close()
    return x

if __name__ == "__main__":
    '''
    data = numpy.asarray(readInputfile(sys.argv[1])).T
    print data
    numpy.savetxt(sys.argv[2], data, fmt="%d")
    print data[111]
    '''
    #data = numpy.loadtxt(sys.argv[1]).T
    data = numpy.genfromtxt(sys.argv[1], dtype=float, invalid_raise=False, missing_values='NULL', usemask=False, filling_values=2.0)
    data[numpy.isnan(data)]=0
    print data.shape
    print data[0,:]
    
    numpy.savetxt(sys.argv[2], data, fmt="%f")
