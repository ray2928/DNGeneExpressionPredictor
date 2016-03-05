from pylearn2.config import yaml_parse
import numpy as np
import sys
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
import time
import timeit
"""
Script to train models

Basic usage:

	/home/rxpkd/tools/Python-2.7.5/bin/python2.7 train_layers.py ../Data/input.txt ../Data/output_mean.txt

"""

if __name__ == "__main__":
    #cross validation shuffle
    x = np.loadtxt(sys.argv[1]).T
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    y = np.loadtxt(sys.argv[2]).T
    cvx = x[0:100, :]
    cvy = y[0:100, :]
    cv = cross_validation.KFold(len(cvx), n_folds = 5)
    xtraining = []
    xvalidating = []
    ytraining = []
    yvalidating = []
    i=0
    for traincv, testcv in cv:
        xtraining = cvx[traincv]
        print "Size of data: ", xtraining.shape
        xvalidating = cvx[testcv]
        print "Size of data: ", xvalidating.shape
        ytraining = cvy[traincv]
        print "Size of data: ", ytraining.shape
        yvalidating = cvy[testcv]
        print "Size of data: ", yvalidating.shape
        i=i+1
        print i, "iteration"
        if i==1:
            break
    cvx = np.concatenate((xtraining, xvalidating), axis=0)
    cvy = np.concatenate((ytraining, yvalidating), axis=0)
    x = np.concatenate((cvx, x[100:112,:]), axis=0)
    y = np.concatenate((cvy, y[100:112,:]), axis=0)
    print "Size of x data: ", x.shape
    print "Size of y data: ", y.shape
    np.savetxt("../Data/cvinput.txt", x, fmt="%f", delimiter="\t")
    np.savetxt("../Data/cvoutput.txt", y, fmt="%f", delimiter="\t")
    print "===========Finish Loading Data=============="
    #raw_input()
    start = timeit.default_timer()
    layer1_yaml = open('../YAML/dae_layer1.yaml', 'r').read()
    hyper_params_l1 = {
	           'input' : "../Data/cvinput.txt",
			   'output' : "../Data/cvoutput.txt",
			   'batch_size' : 100,
		       'monitoring_batches' : 5,
		       'nhid' : 4000,
		       'max_epochs' : 100,
		       'save_path' : '../PKL'}
    layer1_yaml = layer1_yaml % (hyper_params_l1)
    print layer1_yaml
    train = yaml_parse.load(layer1_yaml)
    print "=================Training First Autoencoder================"
    train.main_loop()

    layer2_yaml = open('../YAML/dae_layer2.yaml', 'r').read()
    hyper_params_l2 = {
	           'input' : "../Data/cvinput.txt",
			   'output' : "../Data/cvoutput.txt",
			   'batch_size' : 100,
               'monitoring_batches' : 5,
               'nvis' : hyper_params_l1['nhid'],
               'nhid' : 2000,
               'max_epochs' : 100,
               'save_path' : '../PKL'}
    layer2_yaml = layer2_yaml % (hyper_params_l2)
    print layer2_yaml
    train = yaml_parse.load(layer2_yaml)
    print "=================Training Second Autoencoder================"
    train.main_loop()

    mlp_yaml = open('../YAML/dae_mlp_layer_linear.yaml', 'r').read()
    hyper_params_mlp = {
	           'input' : "../Data/cvinput.txt",
			   'output' : "../Data/cvoutput.txt",
			   'batch_size' : 10,
	           'max_epochs' : 50,
	           'save_path' : '../PKL'}
    mlp_yaml = mlp_yaml % (hyper_params_mlp)
    print mlp_yaml
    train = yaml_parse.load(mlp_yaml)
    print "=================Training Multilayer Perceptron================"
    train.main_loop()
    print "==================Training End================================="
    stop = timeit.default_timer()
    print "Program running time: ", stop - start
