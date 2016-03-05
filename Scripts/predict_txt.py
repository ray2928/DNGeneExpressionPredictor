#!/usr/bin/env python
# coding: utf-8
"""
Script to predict values using a pkl model file.

This is a configurable script to make predictions.

Basic usage:

   python predict_txt.py experiment_5_best.pkl yeast_genotype_matrix_sorted.txt output.txt --prediction_type regression --output_type float

Optionally it is possible to specify if the prediction is regression or
classification (default is classification). The predicted variables are
integer by default.

"""
from __future__ import print_function

import sys
import os
import argparse
import numpy as np

from pylearn2.utils import serial
from theano import tensor as T
from theano import function
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Launch a prediction from a pkl file"
    )
    parser.add_argument('model_filename',
                        help='Specifies the pkl model file')
    parser.add_argument('test_filename',
                        help='Specifies the csv file with the values to predict')
    parser.add_argument('output_filename',
                        help='Specifies the predictions output file')
    parser.add_argument('--prediction_type', '-P',
                        default="classification",
                        help='Prediction type (classification/regression)')
    parser.add_argument('--output_type', '-T',
                        default="int",
                        help='Output variable type (int/float)')
    parser.add_argument('--has-headers', '-H',
                        dest='has_headers',
                        action='store_true',
                        help='Indicates the first row in the input file is feature labels')
    parser.add_argument('--has-row-label', '-L',
                        dest='has_row_label',
                        action='store_true',
                        help='Indicates the first column in the input file is row labels')
    parser.add_argument('--delimiter', '-D',
                        default=',',
                        help="Specifies the CSV delimiter for the test file. Usual values are \
                             comma (default) ',' semicolon ';' colon ':' tabulation '\\t' and space ' '")
    return parser

def predict(model_path, input_path, output_path, predictionType="classification", outputType="int",
            headers=False, first_col_label=False, delimiter=","):
    """
    Predict from a pkl file.

    Parameters
    ----------
    modelFilename : str
        The file name of the model file.
    testFilename : str
        The file name of the file to test/predict.
    outputFilename : str
        The file name of the output file.
    predictionType : str, optional
        Type of prediction (classification/regression).
    outputType : str, optional
        Type of predicted variable (int/float).
    headers : bool, optional
        Indicates whether the first row in the input file is feature labels
    first_col_label : bool, optional
        Indicates whether the first column in the input file is row labels (e.g. row numbers)
    """

    print("loading model...")

    try:
        model = serial.load(model_path)
    except Exception as e:
        print("error loading {}:".format(model_path))
        print(e)
        return False

    print("setting up symbolic expressions...")

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)

    if predictionType == "classification":
        Y = T.argmax(Y, axis=1)

    f = function([X], Y, allow_input_downcast=True)

    print("loading data and predicting...")

   
    skiprows = 1 if headers else 0

    # x is a numpy array
    x = np.loadtxt(input_path).T
    # validation dataset
    x = x[100:112,:]
    #preprocess data    
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)

    #print(x)
    print("==========================================")
    print("Size of gene variation data: ", x.shape)
    testy = np.loadtxt(output_path).T
    testy = testy[100:112, :]
    print("Size of gene expression data: ", testy.shape)

    if first_col_label:
	    x = x[:,1:]
    #make prediction
    y = f(x)
    print("output size: ", y.shape)
    print("writing predictions...")

    variableType = "%d"
    if outputType != "int":
        variableType = "%f"
	#save test label
	test_path = "/home/rxpkd/rui_project_transfer/Result/testy.txt"
    np.savetxt(test_path, testy, fmt=variableType, delimiter="\t")
    print("Test Data Set saved in",test_path)
	#save estimated outcome
    estimated_path = "/home/rxpkd/rui_project_transfer/Result/estimated.txt"
    print("Estimated outcome saved in", estimated_path)
    np.savetxt(estimated_path, y, fmt=variableType, delimiter="\t")
    np.savetxt("min_max.txt", x, fmt=variableType, delimiter="\t")
    MSE = mean_squared_error(testy, y)
    print("error:", MSE)
    return True
	
if __name__ == "__main__":
    """
    See module-level docstring for a description of the script.
    """
    parser = make_argument_parser()
    args = parser.parse_args()
    ret = predict(args.model_filename, args.test_filename, args.output_filename,
        args.prediction_type, args.output_type,
        args.has_headers, args.has_row_label, args.delimiter)
    if not ret:
        sys.exit(-1)

