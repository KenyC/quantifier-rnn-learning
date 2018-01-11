"""
Copyright (C) 2017 Shane Steinert-Threlkeld

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""
import tensorflow as tf
import numpy as np

import data_gen
import quantifiers


INPUT_FEATURE = 'x'


# TODO: some docs here, noting TF estimator stuff
def lstm_model_fn(features, labels, mode, params):

    # for variable length sequences,
    # see http://danijar.com/variable-sequence-lengths-in-tensorflow/
    def length(data):
        """Gets real length of sequences from a padded tensor.

        Args:
            data: a Tensor, containing sequences

        Returns:
            a Tensor, of shape [data.shape[0]], containing the length
            of each sequence
        """
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    # BUILD GRAPH

    # how big each input will be
    num_quants = len(params['quantifiers'])
    item_size = quantifiers.Quantifier.num_chars + num_quants

    # -- input_models: [batch_size, max_len, item_size]
    input_models = features[INPUT_FEATURE]
    # -- input_labels: [batch_size, num_classes]
    input_labels = labels
    # -- lengths: [batch_size], how long each input really is
    lengths = length(input_models)

    cells = []
    for _ in range(params['num_layers']):
        # TODO: consider other RNN cells?
        cell = tf.contrib.rnn.LSTMCell(params['hidden_size'])
        # dropout
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, state_keep_prob=params['dropout'])
        cells.append(cell)
    multi_cell = tf.contrib.rnn.MultiRNNCell(cells)

    # run on input
    # -- output: [batch_size, max_len, out_size]
    output, state = tf.nn.dynamic_rnn(
        multi_cell, input_models,
        dtype=tf.float64, sequence_length=lengths)

    # TODO: modify to allow prediction at every time step

    # extract output at end of reading sequence
    # -- flat_output: [batch_size * max_len, out_size]
    flat_output = tf.reshape(output, [-1, params['hidden_size']])
    # -- indices: [batch_size]
    output_length = tf.shape(output)[0]
    indices = (tf.range(0, output_length) * params['max_len']
               + (lengths - 1))
    # -- final_output: [batch_size, out_size]
    final_output = tf.gather(flat_output, indices)
    tf.summary.histogram('final output', final_output)

    # make prediction
    # TODO: play with arguments here
    # -- logits: [batch_size, num_classes]
    logits = tf.contrib.layers.fully_connected(
        inputs=final_output,
        num_outputs=params['num_classes'],
        activation_fn=None)

    # -- loss: [batch_size]
    loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=input_labels,
            logits=logits)
    # -- total_loss: scalar
    total_loss = tf.reduce_mean(loss)
    tf.summary.scalar('loss', total_loss)

    # training op
    # TODO: try different optimizers, parameters for it, etc
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(total_loss,
                                  global_step=tf.train.get_global_step())

    # -- probs: [batch_size, num_classes]
    probs = tf.nn.softmax(logits)
    # total accuracy
    # -- prediction: [batch_size]
    prediction = tf.argmax(probs, 1)
    # -- target: [batch_size]
    target = tf.argmax(input_labels, 1)
    # -- correct_prediction: [batch_size]
    correct_prediction = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
    tf.summary.scalar('total accuracy', accuracy)

    # list of metrics for evaluation
    eval_metrics = {'accuracy': tf.metrics.accuracy(target, prediction)}

    # metrics by quantifier
    # -- flat_inputs: [batch_size * max_len, item_size]
    flat_input = tf.reshape(input_models, [-1, item_size])
    # -- final_inputs: [batch_size, item_size]
    final_inputs = tf.gather(flat_input, indices)
    # extract the portion of the input corresponding to the quantifier
    # -- quants_by_seq: [batch_size, num_quants]
    quants_by_seq = tf.slice(final_inputs,
                             [0, quantifiers.Quantifier.num_chars],
                             [-1, -1])
    # index, in the quantifier list, of the quantifier for each data point
    # -- quant_indices: [batch_size]
    quant_indices = tf.to_int32(tf.argmax(quants_by_seq, 1))
    # -- prediction_by_quant: a list num_quants long
    # -- prediction_by_quant[i]: Tensor of predictions for quantifier i
    prediction_by_quant = tf.dynamic_partition(
            prediction, quant_indices, num_quants)
    # -- target_by_quant: a list num_quants long
    # -- target_by_quant[i]: Tensor containing true for quantifier i
    target_by_quant = tf.dynamic_partition(
            target, quant_indices, num_quants)
    # -- loss_by_quant: a list num_quants long
    # -- loss_by_quant[i]: Tensor containing loss for quantifier i
    loss_by_quant = tf.dynamic_partition(
            loss, quant_indices, num_quants)

    # TODO: refactor this for eval_metric_ops
    quant_accs = []
    quant_label_dists = []
    quant_loss = []
    for idx in range(num_quants):
        # -- quant_accs[idx]: accuracy for each quantifier
        quant_accs.append(
                tf.reduce_mean(tf.to_float(
                    tf.equal(
                        prediction_by_quant[idx], target_by_quant[idx]))))
        quant_loss.append(
                tf.reduce_mean(loss_by_quant[idx]))
        tf.summary.scalar(
                '{} accuracy'.format(params['quantifiers'][idx]._name),
                quant_accs[idx])
        tf.summary.scalar(
                '{} loss'.format(params['quantifiers'][idx]._name),
                quant_loss[idx])
        _, _, label_counts = tf.unique_with_counts(target_by_quant[idx])
        quant_label_dists.append(label_counts)

    # write summary data
    # summaries = tf.summary.merge_all()
    # test_writer = tf.summary.FileWriter(write_dir, sess.graph)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
        predictions={'probs': probs},
        eval_metric_ops=eval_metrics)


def run_trial(eparams, hparams, trial_num,
              write_dir='/tmp/tensorflow/quantexp', stop_loss=0.01):

    # BUILD MODEL
    model = tf.estimator.Estimator(model_fn=lstm_model_fn, params=hparams)

    # GENERATE DATA
    generator = data_gen.DataGenerator(
            hparams['max_len'], hparams['quantifiers'],
            mode=eparams['generator_mode'],
            num_data_points=eparams['num_data'])

    training_data = generator.get_training_data()
    test_data = generator.get_test_data()

    # TODO: document
    def get_np_data(data):
        x_data = np.array([datum[0] for datum in data])
        y_data = np.array([datum[1] for datum in data])
        return x_data, y_data

    train_x, train_y = get_np_data(training_data)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE: train_x},
        y=train_y,
        batch_size=eparams['batch_size'],
        num_epochs=eparams['num_epochs'],
        shuffle=True)

    test_x, test_y = get_np_data(test_data)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE: test_x},
        y=test_y,
        batch_size=len(test_x),
        shuffle=False)

    model.train(input_fn=train_input_fn, steps=200)
    model.evaluate(input_fn=eval_input_fn)
    model.train(input_fn=train_input_fn, steps=200)
    model.evaluate(input_fn=eval_input_fn)


"""
    tf.reset_default_graph()

    with tf.Session() as sess, tf.variable_scope('trial_' + str(trial_num)) as scope:


        test_models = [datum[0] for datum in test_data]
        test_labels = [datum[1] for datum in test_data]

        # TRAIN

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # TODO: document this and section above that generates the ops
        # measures percentage of models with the same truth value
        # for every quantifier
        label_dists = sess.run(quant_label_dists,
                               {input_models: test_models,
                                input_labels: test_labels})
        for idx in range(len(label_dists)):
            print '{}: {}'.format(
                eparams['quantifiers'][idx]._name,
                float(max(label_dists[idx])) / sum(label_dists[idx]))
            print '{}: {}'.format(
                eparams['quantifiers'][idx]._name,
                sum(label_dists[idx]))

        batch_size = eparams['batch_size']
        accuracies = []

        for epoch_idx in range(eparams['num_epochs']):

            # get training data each epoch, randomizes order
            training_data = generator.get_training_data()
            models = [data[0] for data in training_data]
            labels = [data[1] for data in training_data]

            num_batches = len(training_data) / batch_size

            for batch_idx in range(num_batches):

                batch_models = (models[batch_idx*batch_size:
                                       (batch_idx+1)*batch_size])
                batch_labels = (labels[batch_idx*batch_size:
                                       (batch_idx+1)*batch_size])

                sess.run(train_step,
                         {input_models: batch_models,
                          input_labels: batch_labels})

                if batch_idx % 10 == 0:
                    summary, acc, loss = sess.run(
                        [summaries, accuracy, total_loss],
                        {input_models: test_models, input_labels: test_labels})
                    test_writer.add_summary(summary,
                                            batch_idx + num_batches*epoch_idx)
                    accuracies.append(acc)
                    print 'Accuracy at step {}: {}'.format(batch_idx, acc)

                    # END TRAINING
                    # 1) very low loss, 2) accuracy convergence
                    if loss < stop_loss:
                        return
                    if batch_idx > 100 or epoch_idx > 0:
                        recent_accs = accuracies[-100:]
                        recent_avg = sum(recent_accs) / len(recent_accs)
                        if recent_avg > 0.99:
                            return

            epoch_loss, epoch_accuracy = sess.run(
                    [total_loss, accuracy],
                    {input_models: test_models, input_labels: test_labels})
            print 'Epoch {} done'.format(epoch_idx)
            print 'Loss: {}'.format(epoch_loss)
            print 'Accuracy: {}'.format(epoch_accuracy)
"""


# RUN AN EXPERIMENT

def experiment_one_a(write_dir='data/exp1a'):

    eparams = {'num_epochs': 4, 'batch_size': 8,
               'generator_mode': 'g', 'num_data': 100000}
    hparams = {'hidden_size': 12, 'num_layers': 2, 'max_len': 20,
               'num_classes': 2, 'dropout': 1.0,
               'quantifiers': [quantifiers.at_least_n(4),
                               quantifiers.at_least_n_or_at_most_m(6, 2)]}
    num_trials = 30

    for idx in range(num_trials):
        run_trial(eparams, hparams, idx, write_dir)


def experiment_one_b(write_dir='data/exp1b'):

    eparams = {'num_epochs': 4, 'batch_size': 8,
               'generator_mode': 'g', 'num_data': 100000}
    hparams = {'hidden_size': 12, 'num_layers': 2, 'max_len': 20,
               'num_classes': 2, 'dropout': 1.0,
               'quantifiers': [quantifiers.at_most_n(3),
                               quantifiers.at_least_n_or_at_most_m(6, 2)]}
    num_trials = 30

    for idx in range(num_trials):
        run_trial(eparams, hparams, idx, write_dir)


def experiment_one_c(write_dir='data/exp1c'):

    eparams = {'num_epochs': 4, 'batch_size': 8,
               'generator_mode': 'g', 'num_data': 100000}
    hparams = {'hidden_size': 12, 'num_layers': 2, 'max_len': 20,
               'num_classes': 2, 'dropout': 1.0,
               'quantifiers': [quantifiers.at_least_n(4),
                               quantifiers.between_m_and_n(6, 10)]}
    num_trials = 30

    for idx in range(num_trials):
        run_trial(eparams, hparams, idx, write_dir)


def experiment_one_d(write_dir='data/exp1d'):

    eparams = {'num_epochs': 4, 'batch_size': 8,
               'generator_mode': 'g', 'num_data': 100000}
    hparams = {'hidden_size': 12, 'num_layers': 2, 'max_len': 20,
               'num_classes': 2, 'dropout': 1.0,
               'quantifiers': [quantifiers.at_most_n(4),
                               quantifiers.between_m_and_n(6, 10)]}
    num_trials = 30

    for idx in range(num_trials):
        run_trial(eparams, hparams, idx, write_dir)


def experiment_two(write_dir='data/exp2'):

    eparams = {'num_epochs': 4, 'batch_size': 8,
               'generator_mode': 'g', 'num_data': 200000}
    hparams = {'hidden_size': 12, 'num_layers': 2, 'max_len': 20,
               'num_classes': 2, 'dropout': 1.0,
               'quantifiers': [quantifiers.first_n(3),
                               quantifiers.at_least_n(3)]}
    num_trials = 30

    for idx in range(num_trials):
        run_trial(eparams, hparams, idx, write_dir)


def experiment_three(write_dir='data/exp3'):

    eparams = {'num_epochs': 4, 'batch_size': 8,
               'generator_mode': 'g', 'num_data': 300000}
    hparams = {'hidden_size': 12, 'num_layers': 2, 'max_len': 20,
               'num_classes': 2, 'dropout': 1.0,
               'quantifiers': [quantifiers.nall, quantifiers.notonly]}
    num_trials = 30

    for idx in range(num_trials):
        run_trial(eparams, hparams, idx, write_dir)
