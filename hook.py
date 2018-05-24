import tensorflow as tf
from collections import defaultdict
import util

class NGenStopHook(tf.train.SessionRunHook):
    """Stop after N batches have been observed.

    Writes output of a trial as CSV file."""

    def __init__(self, estimator, eval_input, filename, max_generation,
                 num_steps=50, trialN = 0):

        self._estimator = estimator
        self._input_fn = eval_input
        self._max_generation = max_generation
        self._num_steps = num_steps
        self._trialN=trialN
        # store results of evaluations
        self._results = defaultdict(list)
        self._filename = filename

    def begin(self):

        self._global_step_tensor = tf.train.get_or_create_global_step()
        if self._global_step_tensor is None:
            raise ValueError("global_step needed for EvalEarlyStop")

    def before_run(self, run_context):

        requests = {'global_step': self._global_step_tensor}
        return tf.train.SessionRunArgs(requests)

    def after_run(self, run_context, run_values):

        global_step = run_values.results['global_step']
        if (global_step-1) % self._num_steps == 0:
            ev_results = self._estimator.evaluate(input_fn=self._input_fn)

            print('')
            print('Trial: {} , step {}'.format(self._trialN, global_step))
            for key, value in ev_results.items():
                self._results[key].append(value)
                print('{}: {}'.format(key, value))

        if  global_step > self._max_generation:
            print("##I finished normally##")
            run_context.request_stop()

    def end(self, session):
        # write results to csv
        util.dict_to_csv(self._results, self._filename)