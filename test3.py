# import quantifiers

# eparams = {'num_epochs': 4, 'batch_size': 8,
# 	   'generator_mode': 'g', 'num_data': 300000,
# 	   'eval_steps': 50, 'max_generation': 3000}
# hparams = {'hidden_size': 12, 'num_layers': 2, 'max_len': 20,
# 		   'num_classes': 2, 'dropout': 1.0,
# 		   'quantifiers': [quantifiers.nall]}

# generator = DataGenerator(
#             hparams['max_len'], hparams['quantifiers'],
#             mode=eparams['generator_mode'],
#             num_data_points=eparams['num_data'])

from quantifiers import *

measure_monotonicity(at_least_three)
