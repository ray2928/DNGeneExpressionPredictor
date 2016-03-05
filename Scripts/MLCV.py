from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np
import sys
import timeit

class MLCV(object):
    def __init__(self, train, target, folds):
		print "folds: ", folds
		#read in  data, parse into training and target sets
		print "\n ------------------Load file --------------- \n"
		train = np.loadtxt(input).T
		#replace with method you prefer
		min_max_scaler = preprocessing.MinMaxScaler()
		train = min_max_scaler.fit_transform(train)
		print "Size of read data: ", train.shape
		
		#train = imputation_missingValue(train)

		print "Training data after preprocess:"
		print train
	  
		target = np.loadtxt(target).T
		print "Size of read data: ", target.shape
		self.train = train
		self.target = target
		self.folds = folds
		
	def crossValidate(self, train, folds):
		#K-Fold cross validation.
		cv = cross_validation.KFold(len(self.train), folds, indices=False)
		
		return cv



def imputation_missingValue(input):
    imp = Imputer(missing_values=2, strategy='median', axis=0)
    imp.fit(input)
    return imp.transform(input)

def main(folds = 5):
    print "folds: ", folds
    #read in  data, parse into training and target sets
    print "\n ------------------Load file --------------- \n"
    train = np.loadtxt(sys.argv[1]).T
	#replace with method you prefer
    min_max_scaler = preprocessing.MinMaxScaler()
    train = min_max_scaler.fit_transform(train)
    xtest = train[100:112, :]
    train = train[0:100, :]
    print "Size of read data: ", train.shape
	
    #train = imputation_missingValue(train)

    print "Training data after preprocess:"
    print train
  
    target = np.loadtxt(sys.argv[2]).T
    ytest = target[100:112, :]
    target = target[0:100,:]
    print "Size of read data: ", target.shape

    #estimators = 70
    #print "estimators:", estimators
    #rf = RandomForestRegressor(n_estimators=estimators)

    #Simple K-Fold cross validation.
    cv = cross_validation.KFold(len(train), folds, indices=False)
    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    i = 0
    for traincv, testcv in cv:
        start = timeit.default_timer()
        i += 1
        print i, "epoch"
        #rf.fit(train[traincv], target[traincv])
        #prediction = rf.predict(train[testcv])
        MSE = mean_squared_error(target[testcv], prediction)
        print "MSE: ", MSE, " for ", i, " iteration."
        results.append(MSE)
        stop = timeit.default_timer()
	print "Program running time: ", stop - start 
    #print out the mean of the cross-validated results
    print "Results: " + str( np.array(results).mean() ), "for folds: ", folds
    print "Results for independent data: ", mean_squared_error(rf.fit(train[best_train], target[best_train]).predict(xtest), xtest)

if __name__=="__main__":
    main()
