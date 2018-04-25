from data_gen import DataGenerator
import itertools
import math
import os
import time

from collections import defaultdict

import numpy as np

import quantifiers



class BiasedDG(DataGenerator):
	def __init__(self, max_len, quants=quantifiers.get_all_quantifiers(), probaRestrictor = 0.2, probaScope = 0.2,
				 training_split=0.7, file_path='/tmp/quantexp/data/',
				 bin_size=1e6, num_data_points=100000, min_len=1):
		

		# Maximum size of model
		self._max_len = max_len
		self._min_len = min_len
		# Quantifiers used in the experiment
		self._quantifiers = quants
		self._num_quants = len(quants)
		
		self._pA = probaRestrictor
		self._pB = probaScope

		self._quant_labels = np.identity(self._num_quants)

		# Ratio training/eval data set
		self._training_split = training_split
		self._training_data = None
		self._test_data = None

		self._labeled_data = self._generate_labeled_data(num_data_points)
	
	def _generate_random_tuple(self):
		"""Generates a random tuple corresponding to an input example.

		Returns:
			a pair seq, quant, where seq is a random sequence of characters
			of a random length up to self._max_len and quant is a random
			integer up to self._num_quants
		"""
		quant = np.random.randint(self._num_quants)
		length = np.random.randint(self._min_len, self._max_len+1)
		seq = tuple((BiasedDG._biased_randint_gen(self._pA, self._pB)
					 for _ in range(length)))
		return seq, quant

	def _biased_randint_gen(pA=0.5,pB=0.5):
		return np.random.choice(quantifiers.Quantifier.num_chars,p=[pA*pB,pA*(1-pB),(1-pA)*pB,(1-pA)*(1-pB)])

class BiasedDGNAllNOnly(BiasedDG):
	# We optimize the data generation process for not only/not all quantifiers

	def __init__(self, max_len, quants=quantifiers.get_all_quantifiers(), probaRestrictor = 0.2, probaScope = 0.2,
				 training_split=0.7, file_path='/tmp/quantexp/data/',
				 bin_size=1e6, num_data_points=100000, min_len=1):
		
		

		self._count = True
		
		self._pA = probaRestrictor
		self._pB = probaScope


		self._arrBis = np.array([self._pA*self._pB,self._pA*(1-self._pB),(1-self._pA)*(1-self._pB)])
		self._arrBis = self._arrBis/np.sum(self._arrBis)
		self._arrTer = np.array([self._pA*self._pB,(1-self._pA)*self._pB,(1-self._pA)*(1-self._pB)])
		self._arrTer = self._arrTer/np.sum(self._arrTer)

		super(BiasedDGNAllNOnly, self).__init__(max_len, [quantifiers.nall,quantifiers.notonly], probaRestrictor, probaScope, training_split, file_path,
				 bin_size, num_data_points, min_len)
		
	
	def _generate_random_tuple(self):


		quant = np.random.randint(self._num_quants)
		length = np.random.randint(self._min_len, self._max_len+1)
		
		if self._count and self._quantifiers[quant]._name=="not_only":
			seq = tuple((self._biased_randint_gen_bis()
							 for _ in range(length)))
		elif self._count and self._quantifiers[quant]._name=="not_all":
			seq = tuple((self._biased_randint_gen_bis()
							 for _ in range(length)))
		else:
		# if True:
			seq = tuple((BiasedDG._biased_randint_gen(self._pA,self._pB)
							 for _ in range(length)))
		
		self._count=not self._count
		return seq, quant

	def _biased_randint_gen_bis(self):
		return [0,1,3][np.random.choice(3,p=self._arrBis)]

	def _biased_randint_gen_ter(self):
		return [0,2,3][np.random.choice(3,p=self._arrTer)]