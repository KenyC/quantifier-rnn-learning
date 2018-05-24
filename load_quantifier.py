from collections import defaultdict
import tensorflow as tf
import numpy as np

import data_gen
import csv
import bias_data_gen
import quantifiers
import util
import quant_verify as expe
from randDataGen import RandomDG
from hook import NGenStopHook
from quantifier_util import QuantifierFromNN


def loadQuantNN(folder):
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=50,
        save_checkpoints_secs=None,
        save_summary_steps=50)

    model = tf.estimator.Estimator(
        model_fn=expe.lstm_model_fn,
        params={'hidden_size': 12, 'num_layers': 2, 'max_len': 20,
         'num_classes': 2, 'dropout': 1.0,
         'quantifiers': [quantifiers.first_three]},
        model_dir=folder,
        config=run_config)

    return QuantifierFromNN(model)



gen=[0,0,0]
testSeq=np.array([quantifiers.Quantifier.chars[x] for x in gen]*2)
