import numpy as np
import itertools
import math
import quantifiers

#TODO: move batching logic from quant_verify.run_experiment to here?
#TODO: allow reading/writing data to files instead of in memory?

class DataGenerator(object):

    #TODO: document; mode = r, w, g [generate]
    def __init__(self, max_len, quants=quantifiers.get_all_quantifiers(), training_split=0.7,
            mode='r', file_path='/tmp/quantexp/data/', test_bins=12):

        self._max_len = max_len
        self._quantifiers = quants
        self._quant_labels = np.identity(len(quants))
        self._training_split = training_split
        self._training_data = None
        self._test_data = None

        if mode == 'g':
            self._labeled_data = self._generate_labeled_data()

    def _generate_sequences(self):
        """Generates (sequence, quantifier_index) pairs for all sequences
        up to length max_len.
        These correspond to finite models. 

        Args:
            max_len: the maximum length of a sequence (aka size of a model)

        Returns: 
            a generator, generating all relevant pairs
        """

        num_quants = len(self._quantifiers)
        num_chars = quantifiers.Quantifier.num_chars

        all_gens = []
        for n in xrange(1, self._max_len + 1):
            seqs = itertools.product(xrange(num_chars), repeat=n)
            data_n = ( (seq, quant) for seq in seqs for quant in xrange(num_quants) )
            all_gens.append(data_n)

        return itertools.chain(*all_gens)

    def _point_from_tuple(self, tup):
        """Generates a labeled data point from a tuple generated by _generate_sequences.
        To do so, it converts character indices into one-hot vectors, pads the length to _max_len,
        and augments each character with the one-hot vector corresponding to the quantifier.
        It then runs the quantifier on the sequence and outputs the generated label as well.

        Args:
            tup: a pair, the first element of which is a tuple of elements of range(num_chars), 
            the second element of which is an element of range(num_quants)

        Returns:
            a pair, the first element of which is a max_len length tuple of numpy arrays of length
            num_chars + num_quants, corresponding to the characters in the sequence, the second
            element of which is a label, generated by running the quantifier on the input sequence.
        """

        char_seq, quant_idx = tup

        chars = tuple(quantifiers.Quantifier.chars[idx] for idx in char_seq)
        padded_seq = chars + (quantifiers.Quantifier.zero_char,)*(self._max_len - len(chars))
        padded_with_quant = tuple(np.concatenate([char, self._quant_labels[quant_idx]]) for char in padded_seq)
        label = self._quantifiers[quant_idx](chars)

        return padded_with_quant, label

    def _generate_labeled_data(self):
        """Generates a complete list of labeled data.  Iterates through
        _generate_sequences, calling _point_from_tuple on each tuple generated.
        At the end, the list is shuffled so that the data is in random order.
        Note that this returns the entire dataset, not split into train/test sets.

        Returns:
            a list of all labeled data, in random order.
        """

        self._labeled_data = []

        for tup in self._generate_sequences():
            self._labeled_data.append(
                    self._point_from_tuple(tup) )

        np.random.shuffle(self._labeled_data)
        return self._labeled_data

    def get_training_data(self):
        """Gets training data, based on the percentage self._training_split.
        Shuffles the training data every time it is called.
        Must be called only after _generate_labeled_data has been.
        """

        if self._training_data is None:
            idx = int(math.ceil(self._training_split * len(self._labeled_data)))
            self._training_data = self._labeled_data[:idx]
            
        np.random.shuffle(self._training_data)
        return self._training_data

    def get_test_data(self):
        """Gets test data, based on the percentage 1 - self._training_split.
        Must be called only after _generate_labeled_data has been.
        """

        if self._test_data is None:
            idx = int(math.ceil(self._training_split * len(self._labeled_data)))
            self._test_data = self._labeled_data[idx:]

        return self._test_data
