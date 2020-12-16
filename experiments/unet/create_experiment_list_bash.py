import itertools
import argparse
import json
import os

'''
TODO:
	* determine class weights
	* awd

'''


parser = argparse.ArgumentParser(description='Create experiment definitions')
parser.add_argument("-output_filepath", help="output file for experiment definitions", type=str, required=True)
parser.add_argument("-run_dir", help="output path for experiment results", type=str, required=True)

run_params = parser.parse_args()
run_params = vars(run_params)
output_filepath = run_params['output_filepath']
run_dir = run_params['run_dir']
assert len(output_filepath) > 0, 'empty output filepath'
assert len(run_dir) > 0, 'empty run dir'


# param_ranges = dict(
# 	accumulate_batches=[1,8,16],
# 	# weight_decay=[0.01, 0.001, 0.0001],
# 	loss_function = [
# 		'cross_entropy',
# 		'soft_dice',
# 		('focal_loss', ('gamma', 1)),
# 		('focal_loss', ('gamma', 2)),
# 		('focal_loss', ('gamma', 5)),		
# 		('weighted_cross_entropy', ('class_weights', [0.52,0.85,1.09,1.28,1.26])),
# 		('weighted_cross_entropy', ('class_weights', [0.25,0.67,1.10,1.51,1.48])),
# 		('weighted_cross_entropy', ('class_weights', [0.05,0.36,0.98,1.84,1.77]))
# 	],
# 	lr = [0.01, 0.001]
# )

param_ranges = dict(
	augmentation_brightness=[1],
	augmentation_contrast=[1],
	augmentation_rotate3d=[1,2,3],
	nepochs=[150]
)




experiments = []
for i, res in enumerate(itertools.product(*param_ranges.values())):
	param_dict = {}
	for k, v in zip(param_ranges.keys(),res):
		if isinstance(v, tuple):
			param_dict[k] = v[0]
			other_param_name, other_param_value = v[1]
			param_dict[other_param_name] = other_param_value
		else:
			param_dict[k] = v
	param_dict['out_dir'] = os.path.join(run_dir, f"experiment_{i}")
	experiments.append({'experiment': i, 'config_updates': param_dict})


with open(output_filepath, 'w') as f:
	for experiment in experiments:
		command_config_updates = " ".join([f"'{k}={json.dumps(v, separators=(',', ':'))}'" for k,v in experiment['config_updates'].items()])		
		bash_command = f"python train_new_debug.py --name experiment_{experiment['experiment']} with {command_config_updates}"
	
		f.write(bash_command + '\n')
	# f.write(json.dumps(experiments, indent=4))

