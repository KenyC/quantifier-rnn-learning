import tensorflow as tf
import numpy as np
from quantifiers import *

class QuantifierFromNN(Quantifier):
	"""docstring for QuantifierFromNN"""
	def __init__(self, NeuralNetwork, name = "NN-based"):
		super(QuantifierFromNN, self).__init__(name,fn=self.evaluate)
		self._NN = NeuralNetwork
	
	def evaluate(self, seq):
		return self.evaluateMultiple([seq])[0]

	def evaluateMultiple(self, seqs):
		max_len = self._NN.params['max_len']
		n = len(seqs)

		inputSeq = np.full((n,max_len,Quantifier.num_chars+1),0.0)

		for idx in range(n):
			seq = seqs[idx]
			seq = np.concatenate([seq,
				np.full((max_len-len(seq),Quantifier.num_chars),0.0)
				])
			inputSeq[idx] = np.concatenate([seq,
				np.full((len(seq),1),1.0)],
				axis=1)

		predict_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": inputSeq},
			shuffle=False)
		
		predictions = list(self._NN.predict(input_fn=predict_input_fn))

		return [Quantifier.T if p["probs"][0]>p["probs"][1] else Quantifier.F for p in predictions]

class RandomQuantifier(Quantifier):
	def __init__(self, name = "RandomLabeller"):
		super(RandomQuantifier, self).__init__(name,fn=self.evaluate)
	
	def evaluate(self, seq):
		return Quantifier.F if np.random.randint(2)==0 else Quantifier.T