import tensorflow as tf
import itertools as iter
import random
import math

### DATA GENERATION ###

def generate_all_seqs(length, shuffle=True):
    seqs = list(iter.product([0,1], repeat=length))
    if shuffle:
        random.shuffle(seqs)
    return seqs

def at_least_three(seq):
    # we return [0,1] for True and [1,0] for False
    return [0,1] if sum(seq) >= 3 else [1,0]

def get_labeled_data(seqs, func):
    return seqs, [func(seq) for seq in seqs]

# generate all labeled data
SEQ_LEN = 16
NUM_CLASSES = 2
TRAIN_SPLIT = 0.8

X, Y = get_labeled_data(generate_all_seqs(SEQ_LEN), at_least_three)

# split into training and test sets
pivot_index = int(math.ceil(TRAIN_SPLIT*len(X)))

trainX, trainY = X[:pivot_index], Y[:pivot_index]
testX, testY = X[pivot_index:], Y[pivot_index:]
print(len(trainX))

### Feed-Forward Neural Network

class FFNN(object):
    
    def __init__(self, input_size, output_size, hidden_size=10):
        
        # first, basic network architecture
        
        # -- inputs: [batch_size, input_size]
        inputs = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
        self._inputs = inputs
        # -- labels: [batch_size, output_size]
        labels = tf.placeholder(shape=[None, output_size], dtype=tf.float32)
        self._labels = labels
        
        # we will have one hidden layer
        # in general, this should be parameterized
        
        # -- weights1: [input_size, hidden_size]
        weights1 = tf.Variable(tf.random_uniform(shape=[input_size, hidden_size]))
        # -- biases1: [hidden_size]
        biases1 = tf.Variable(tf.random_uniform(shape=[hidden_size]))
        # -- linear: [batch_size, hidden_size]
        linear = tf.add(tf.matmul(inputs, weights1), biases1)
        # -- hidden: [batch_size, hidden_size]
        hidden = tf.nn.relu(linear)
        
        # -- weights2: [hidden_size, output_size]
        weights2 = tf.Variable(tf.random_uniform(shape=[hidden_size, output_size]))
        # -- biases2: [output_size]
        biases2 = tf.Variable(tf.random_uniform(shape=[output_size]))
        # -- logits: [batch_size, output_size]
        logits = tf.add(tf.matmul(hidden, weights2), biases2)
        
        # second, define loss and training
        # -- cross_entropy: [batch_size]
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits)
        # -- loss: []
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer()
        self._train_op = optimizer.minimize(loss)
        
        # finally, some evaluation ops
        
        # -- probabilities: [batch_size, output_size]
        # Probabilities of the sentence being true
        probabilities = tf.nn.softmax(logits)
        self._probabilities = probabilities
        # -- predictions: [batch_size]
        predictions = tf.argmax(probabilities, axis=1)
        # -- targets: [batch_size]
        targets = tf.argmax(labels, axis=1)
        falseT = tf.subtract(tf.constant(1, dtype='int64'),targets)
        # -- correct_prediction: [batch_size]
        correct_prediction = tf.equal(predictions, targets)
        # -- numbers : []
        nFalseTargets = tf.reduce_sum(tf.to_float(falseT))
        nCorrectOnFalse = tf.reduce_sum(tf.to_float(tf.boolean_mask(correct_prediction,falseT)))
        accuracyOnTrue = tf.reduce_mean(tf.to_float(tf.boolean_mask(correct_prediction,targets)))
        accuracyOnFalse = tf.reduce_mean(tf.to_float(tf.boolean_mask(correct_prediction,falseT)))
        #accuracyOnFalse = tf.reduce_mean(tf.to_float(tf.boolean_mask(correct_prediction,tf.toBool(tf.cons-targets))))


        # -- accuracy: []
        accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
        # more evaluation ops could be added here
        self._eval_dict = {
            'accuracy': accuracy,
            'accuracyOnTrue': accuracyOnTrue,
            'accuracyOnFalse': accuracyOnFalse,
            'Num False Targets': nFalseTargets,
            'Num Correct On False': nCorrectOnFalse
        }
        
    @property
    def train(self):
        return self._train_op
    
    @property
    def predictions(self):
        return self._probabilities
    
    @property
    def evaluate(self):
        return self._eval_dict
    
    @property
    def inputs(self):
        return self._inputs
    
    @property
    def labels(self):
        return self._labels

### TRAINING ### 




tf.reset_default_graph()

with tf.Session() as sess:

    # build our model
    model = FFNN(SEQ_LEN, NUM_CLASSES)
    # initialize the variables
    sess.run(tf.global_variables_initializer())
    
    # MAIN TRAINING LOOP
    NUM_EPOCHS = 5
    BATCH_SIZE = 12
    num_batches = len(trainX) // BATCH_SIZE
    
    for epoch in range(NUM_EPOCHS):
        
        # shuffle the training data at start of each epoch
        train_data = list(zip(trainX, trainY))
        random.shuffle(train_data)
        trainX = [datum[0] for datum in train_data]
        trainY = [datum[1] for datum in train_data]
        
        for batch_idx in range(num_batches):
            # get batch of training data
            batchX = trainX[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
            batchY = trainY[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
            # train on the batch
            sess.run(model.train, 
                     {model.inputs: batchX,
                      model.labels: batchY})
            
            # evaluate every N training steps (batches)
            if batch_idx % 1000 == 0:
                print('\nEpoch {}, batch {}, evaluation'.format(epoch, batch_idx))
                print(sess.run(model.evaluate, {model.inputs: testX, model.labels: testY}))


print(trainY.count([0,1]))