from pylearn2.config import yaml_parse
import sys

"""
Script to train models

This is a configurable script to make predictions.

Basic usage:

.. code-block:: none

   python predict_txt.py experiment_5_best.pkl yeast_genotype_matrix_sorted.txt output.txt --prediction_type regression --output_type float

Optionally it is possible to specify if the prediction is regression or
classification (default is classification). The predicted variables are
integer by default.

"""

if __name__ == "__main__":
    
    layer1_yaml = open('../YAML/dae_layer1.yaml', 'r').read()
    hyper_params_l1 = {
	           'input' : sys.argv[1],
			   'output' : sys.argv[2],
			   'batch_size' : 100,
		       'monitoring_batches' : 5,
		       'nhid' : 1500,
		       'max_epochs' : 100,
		       'save_path' : '../PKL'}
    layer1_yaml = layer1_yaml % (hyper_params_l1)
    print layer1_yaml
    train = yaml_parse.load(layer1_yaml)
    print "=================Training First Autoencoder================"
    train.main_loop()
    
    layer2_yaml = open('../YAML/dae_layer2.yaml', 'r').read()
    hyper_params_l2 = {
	           'input' : sys.argv[1],
			   'output' : sys.argv[2],
			   'batch_size' : 100,
               'monitoring_batches' : 5,
               'nvis' : hyper_params_l1['nhid'],
               'nhid' : 750,
               'max_epochs' : 100,
               'save_path' : '../PKL'}
    layer2_yaml = layer2_yaml % (hyper_params_l2)
    print layer2_yaml
    train = yaml_parse.load(layer2_yaml)
    print "=================Training Second Autoencoder================"
    train.main_loop()

    mlp_yaml = open('../YAML/dae_mlp_layer_linear.yaml', 'r').read()
    hyper_params_mlp = {
	           'input' : sys.argv[1],
			   'output' : sys.argv[2],
			   'batch_size' : 10,
	           'max_epochs' : 50,
	           'save_path' : '../PKL'}
    mlp_yaml = mlp_yaml % (hyper_params_mlp)
    print mlp_yaml
    train = yaml_parse.load(mlp_yaml)
    print "=================Training Multilayer Perceptron================"
    train.main_loop()
