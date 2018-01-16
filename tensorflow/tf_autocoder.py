
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("E:\\",one_hot =False)

# parameters 
learning_rate = 0.001
training_epochs = 20
batch_size = 256
display_step =1


# network parameters
num_input = 784

# placehoder
X = tf.placeholder(tf.float32 , [None , num_input])


# hidden layer settings
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2

weights = {
	'encoder_hidden_1': tf.Variables(tf.random_normal([ num_input, n_hidden_1 ])),
	'encoder_hidden_2': tf.Variables(tf.random_normal([ n_hidden_1, n_hidden_2 ])),
	'encoder_hidden_3': tf.Variables(tf.random_normal([ n_hidden_2, n_hidden_3 ])),
	'encoder_hidden_4': tf.Variables(tf.random_normal([ n_hidden_3, n_hidden_4 ])),

	'decoder_hidden_1': tf.Variables(tf.random_normal([ n_hidden_4, n_hidden_3 ])),
	'decoder_hidden_2': tf.Variables(tf.random_normal([ n_hidden_3, n_hidden_2 ])),
	'decoder_hidden_3': tf.Variables(tf.random_normal([ n_hidden_2, n_hidden_1 ])),
	'decoder_hidden_4': tf.Variables(tf.random_normal([ n_hidden_1, num_input ]))
}

biases ={
	'encoder_hidden_1': tf.Variables(tf.random_normal([n_hidden_1 ])),
	'encoder_hidden_2': tf.Variables(tf.random_normal([n_hidden_2 ])),
	'encoder_hidden_3': tf.Variables(tf.random_normal([n_hidden_3 ])),
	'encoder_hidden_4': tf.Variables(tf.random_normal([n_hidden_4 ])),

	'decoder_hidden1': tf.Variables(tf.random_normal([ n_hidden_3 ])),
	'decoder_hidden2': tf.Variables(tf.random_normal([ n_hidden_2 ])),
	'decoder_hidden3': tf.Variables(tf.random_normal([ n_hidden_1 ])),
	'decoder_hidden4': tf.Variables(tf.random_normal([ num_input ]))
}


#bulid the decoder
def encoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_hidden_1']) , biases['encoder_hidden_1']))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_hidden_2']) , biases['encoder_hidden_2']))
	layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_hidden_3']) , biases['encoder_hidden_3']))
	layer_4 = tf.add(tf.matmul(layer_1,weights['encoder_hidden_4']) , biases['encoder_hidden_4'])
	return layer_4

def decoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_hidden_1']) , biases['decoder_hidden_1']))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['decoder_hidden_2']) , biases['decoder_hidden_2']))
	layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_hidden_3']) , biases['decoder_hidden_3']))
	layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['decoder_hidden_4']) , biases['decoder_hidden_4']))
	return layer_2

# bulid the autocoder model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# prediction
y_pred = decoder_op
# target 
y_true = X

# define loss and optimizer
cost = tf.reduce_mean(tf.power(y_true - y_pred , 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# init
init = tf.initialize_all_variables()

# session
with tf.Session() as sess:
	sess.run(init)
	num_batch = int(mnist.train.num_examples/batch_size)

	#training
	for epoch in range(training_epochs):
		for step in range(num_batch):
			batch_x , batch_y = mnist.train.next_batch(batch_size)
			_, cost = sess.run([optimizer , cost], feed_dict = {X : batch_x})

		print('epoch is %d'%(epoch))
	print('optimization is done!')

	#test
	print('testing -----')
	encode_result = sess.run(y_pred , {X:mnist.test.images})
	plt.scatter(encode_result[:,0] , encode_result[:,1] ,c=mnist.test.labels)
	plt.show()