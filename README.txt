Rough README for the Gene Expression Prediction program.
For installing Pylearn2 or other needed packages, please
refer to Intall_pylearn2_guide.txt.

1. Instructions for predicting Gene Expression:
Enter the Scripts/ directory.
Use the predict.py script to predict Gene Expression.

The YAML directory stores the customized YAML for Gene Expression Prediction, anyone can design their model by 
modify the YAML file for differnt problem.

2. To train a model:

Directly give a Genotype File Processed (Example Sample stored in the /home/rxpkd/rui_project_transfer/Data/):

  Usage:
  	python train_layers.py input label

  Example:
	/home/rxpkd/tools/Python-2.7.5/bin/python2.7 train_layers.py ../Data/input.txt  ../Data/output_mean.txt

If autoencoders has been trained, you can just skip the process and just load to pkl files 
stored in /home/rxpkd/rui_project_transfer/PKL, then you just need to train the mlp layer:
  
  Usage:
  	python train_mlp.py input label

  Example:
  	/home/rxpkd/tools/Python-2.7.5/bin/python2.7 train_mlp.py ../Data/input.txt  ../Data/output_mean.txt
  
Note that all model trained before can be found in the /home/rxpkd/rui_project_transfer/PKL.


3. To make a prediction:

  Usage:
	python predict_txt.py model.pkl  input label --prediction_type regression --output_type float

  Example:
	/home/rxpkd/tools/Python-2.7.5/bin/python2.7 predict_txt.py ../PKL/dae_mlp_best.pkl ../Data/input.txt  ../Data/output_mean.txt --prediction_type regression --output_type float

