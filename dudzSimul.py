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


INPUT_FEATURE = 'x'
WRITE_DIR = "atleastsixoratmosttwo/"
PROTO_QUANTIFIER = quantifiers.at_least_n_or_at_most_m(6, 2)
MEASURE = quantifiers.measure_monotonicity
RUN_TRIAL = [0]

expSetup={
    "numGenerations": 40, 
    "dataStop": 20,
    "evalStops": 50,
}



"""
Run one generation of iterated learning 
"""
def run_one_gen(eparams, hparams, trial_num,write_path='/tmp/tensorflow/quantexp'):

    tf.reset_default_graph()

    write_dir = '{}/trial_{}'.format(write_path, trial_num)
    csv_file = '{}/trial_{}.csv'.format(write_path, trial_num)

    # BUILD MODEL
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=eparams['eval_steps'],
        save_checkpoints_secs=None,
        save_summary_steps=eparams['eval_steps'])

    model = tf.estimator.Estimator(
        model_fn=expe.lstm_model_fn,
        params=hparams,
        model_dir=write_dir,
        config=run_config)

    # GENERATE DATA

    if "generator" not in eparams:
        generator = data_gen.DataGenerator(
            hparams['max_len'], hparams['quantifiers'],
            mode=eparams['generator_mode'],
            num_data_points=eparams['num_data'])
    else:
        generator = eparams["generator"]

    training_data = generator.get_training_data()
    test_data = generator.get_test_data()

    def get_np_data(data):
        x_data = np.array([datum[0] for datum in data])
        y_data = np.array([datum[1] for datum in data])
        return x_data, y_data

    # input fn for training
    train_x, train_y = get_np_data(training_data)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE: train_x},
        y=train_y,
        batch_size=eparams['batch_size'],
        num_epochs=eparams['num_epochs'],
        shuffle=True)

    # input fn for evaluation
    test_x, test_y = get_np_data(test_data)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE: test_x},
        y=test_y,
        batch_size=len(test_x),
        shuffle=False)

    print('\n------ TRIAL {} -----'.format(trial_num))

    # train and evaluate model together, using the Hook
    model.train(input_fn=train_input_fn,
                hooks=[NGenStopHook(model, eval_input_fn, csv_file,
                                         eparams['max_generation'], eparams['eval_steps'],
                                         trialN = trial_num)])
    return model




eparams1 = {'num_epochs': 4, 'batch_size': 8,
       'generator_mode': 'g', 'num_data': 300000,
       'eval_steps': 50, 'max_generation': 15000}
hparams1 = {'hidden_size': 12, 'num_layers': 2, 'max_len': 20,
           'num_classes': 2, 'dropout': 1.0,
           'quantifiers': [PROTO_QUANTIFIER]}

for i in RUN_TRIAL:
    write_dir_trial = WRITE_DIR+"trial"+str(i)+"/"
    oi = []
    model = None

    for i in range(expSetup["numGenerations"]):
        model = run_one_gen(eparams1,hparams1, i, write_dir_trial)
        newQ = QuantifierFromNN(model)
        hparams1["quantifiers"][0] = newQ
        oi.append(dict(MEASURE(newQ),generation = i))


    with open(write_dir_trial+"results.csv","w") as f:
        writer = csv.DictWriter(f,oi[0].keys())
        writer.writeheader()
        writer.writerows(oi)

# for i in [1]:
#     write_dir_trial = WRITE_DIR+"trial"+str(i)+"/"
#     oi = []
#     model = None

#     for i in range(expSetup["numGenerations"]):
#       model = run_one_gen(eparams1,hparams1, i, write_dir_trial)
#       newQ = QuantifierFromNN(model)
#       hparams1["quantifiers"][0] = newQ
#       oi.append(dict(quantifiers.measure_order_invariance(newQ),generation = i))


#     with open(write_dir_trial+"results.csv","w") as f:
#       writer = csv.DictWriter(f,oi[0].keys())
#       writer.writeheader()
#       writer.writerows(oi)


