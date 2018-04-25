import tensorflow as tf


# Define NN graphs
weights = tf.Variable(tf.random_uniform(shape=[2, 1]))
inputs = tf.placeholder(shape = [None,2], dtype = tf.float32)
bias = tf.Variable(0.0)
output = tf.nn.sigmoid(tf.add(tf.matmul(inputs,weights),bias))


#Real output variables
realOutput=tf.placeholder(shape = [None,1], dtype= tf.float32)




# Set up cost and learning
cost = tf.reduce_mean(tf.square(output-realOutput))
optimizer = tf.train.AdamOptimizer(1).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(5).minimize(cost)

#Random stuff
W_hist = tf.summary.histogram("weights", weights)
grads = tf.gradients(cost, weights)

# AND data
nEpoch = 100
sizeData=4
dataX= [[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]]*nEpoch
dataY= [[0.0],[0.0],[0.0],[1.0]]*nEpoch



with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	initW=sess.run(weights)
	biasW=sess.run(bias)
	for i in range(nEpoch):
		sess.run(optimizer, {inputs : dataX[i:i+1], realOutput: dataY[i:i+1]})
		if i%10==0:
			print("Epoch %i:"%i)
			#print(sess.run(output,{inputs : dataX[sizeData*i:sizeData*i+4], realOutput: dataY[sizeData*i:sizeData*i+4]}))
			print(sess.run(grads,{inputs : dataX[sizeData*i:sizeData*i+4], realOutput: dataY[sizeData*i:sizeData*i+4]}))

	print("Initial Weights :",initW)
	print("Initial Bias:",biasW)
	print("Weights :",sess.run(weights))
	print("Bias:",sess.run(bias))
	print("Cost:",sess.run(cost,{inputs : dataX[sizeData*i:sizeData*i+4], realOutput: dataY[sizeData*i:sizeData*i+4]}))
	print("Output:",sess.run(output,{inputs : dataX[sizeData*i:sizeData*i+4], realOutput: dataY[sizeData*i:sizeData*i+4]}))