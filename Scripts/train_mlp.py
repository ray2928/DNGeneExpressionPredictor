from pylearn2.config import yaml_parse
import sys

if __name__ == "__main__":
    mlp_yaml = open('../YAML/dae_mlp_layer_linear.yaml', 'r').read()
    hyper_params_mlp = {
					'input' :  sys.argv[1],
					'output' : sys.argv[2],
					'batch_size' : 10,
	                'max_epochs' : 50,
	                'save_path' : '../PKL'}
    mlp_yaml = mlp_yaml % (hyper_params_mlp)
    print mlp_yaml
    train = yaml_parse.load(mlp_yaml)
    train.main_loop()
