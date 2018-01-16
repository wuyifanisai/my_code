
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("E:\\",one_hot =False)

# parameters 
learning_rate = 0.01
training_epochs = 5
batch_size = 256
display_step =1
examples_to_show = 10

# network parameters
num_input = 784

# placehoder
X = tf.placeholder(tf.float32 , [None , num_input])


# hidden layer settings
n_hidden_1 = 256
n_hidden_2 = 128

weights = {
	'encoder_hidden_1': tf.Variables(tf.random_normal([ num_input, n_hidden_1 ])),
	'encoder_hidden_2': tf.Variables(tf.random_normal([ n_hidden_1, n_hidden_2 ])),

	'decoder_hidden1': tf.Variables(tf.random_normal([ n_hidden_2, n_hidden_1 ])),
	'decoder_hidden2': tf.Variables(tf.random_normal([ n_hidden_1, num_input ]))
}

biases ={
	'encoder_hidden_1': tf.Variables(tf.random_normal([n_hidden_1 ])),
	'encoder_hidden_2': tf.Variables(tf.random_normal([n_hidden_2 ])),

	'decoder_hidden1': tf.Variables(tf.random_normal([ n_hidden_1 ])),
	'decoder_hidden2': tf.Variables(tf.random_normal([ num_input ]))
}


#bulid the decoder
def encoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_hidden_1']) , biases['encoder_hidden_1']))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_hidden_2']) , biases['encoder_hidden_2']))
	return layer_2

def decoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_hidden_1']) , biases['decoder_hidden_1']))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['decoder_hidden_2']) , biases['decoder_hidden_2']))
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
	encode_decoder = sess.run(y_pred , {X:mnist.test.images[:examples_to_show]})
	f,a = plt.subplots(2,10 , figsize = (10,2))
	for i in range(examples_to_show):
		a[0][i].imshow(np.reshape(mnist.test.images[i],(28 ,28)))
		a[1][i].imshow(np.reshape(encode_decoder[i],(28 ,28)))
	plt.show()
