import numpy as np
from data_gen import DataGenerator
import quantifiers
from quantifier_util import RandomQuantifier


class RandomDG(DataGenerator):
	def __init__(self, max_len,training_split=0.7, mode='g', file_path='/tmp/quantexp/data/',
				 bin_size=1e6, num_data_points=100000, min_len=1):
		self._max_len = max_len
		self._min_len = min_len
		# Quantifiers used in the experiment
		self._quantifiers = [RandomQuantifier()]
		self._num_quants = 1
		

		self._quant_labels = np.identity(self._num_quants)

		# Ratio training/eval data set
		self._training_split = training_split
		self._training_data = None
		self._test_data = None

		self._labeled_data = self._generate_labeled_data(num_data_points)
	