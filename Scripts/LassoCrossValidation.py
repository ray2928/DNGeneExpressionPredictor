from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from scipy import stats
import numpy as np
import sys
import timeit

def imputation_missingValue(input):
    imp = Imputer(missing_values=2, strategy='median', axis=0)
    imp.fit(input)
    return imp.transform(input)
def main(folds = 5):
    print "folds: ", folds
    #read in  data, parse into training and target sets
    print "\n ------------------Load file --------------- \n"
    train = np.loadtxt(sys.argv[1]).T
    min_max_scaler = preprocessing.MinMaxScaler()
    train = min_max_scaler.fit_transform(train)
	#test data set
    xtest = train[100:112, :]
    train = train[0:100, :]
    print "Size of read data: ", train.shape
    #train = imputation_missingValue(train)
    print "After Standardization:"
    print train
  
    target = np.loadtxt(sys.argv[2]).T
    ytest = target[100:112, :]
    target = target[0:100,:]
    print "Size of read data: ", target.shape

    al = 0.3
    rf = Lasso(alpha=al)
	
    #Simple K-Fold cross validation.
    cv = cross_validation.KFold(len(train), folds)
    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    i = 0
    min_MSE = sys.maxint
    best_train = -1
    best_test = -1
    for traincv, testcv in cv:
        start = timeit.default_timer()
        i += 1
        print i, "epoch"
        rf.fit(train[traincv], target[traincv])
        prediction = rf.predict(train[testcv])
        MSE = mean_squared_error(target[testcv], prediction)
        print "MSE: ", MSE, " for ",i
        if min_MSE > MSE:
            best_train = traincv
            best_test = testcv
            min_MSE = MSE
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        results.append(MSE)
        stop = timeit.default_timer()
	print "Program running time: ", stop - start 
    #print out the mean of the cross-validated results
    print "Results: " + str( np.array(results).mean() ), "for folds: ", folds
    print "Results for independent data: ", mean_squared_error(rf.fit(train[best_train], target[best_train]).predict(xtest), ytest)
    print "R squared:"
    print "alpha:", al

if __name__=="__main__":
    main()
