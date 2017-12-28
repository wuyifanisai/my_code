'''
highway deep networks
'''
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

total_layers = 50
number_units = total_layers/5

####################### define highway unit ########################
def highwayUnit(input_layer, id):
	with tf.variable('highway'+str(id)):

		H = slim.conv2d(input_layer, 64, [3,3])

		T = slim.conv2d(
						input_layer ,64, [3.3],
			            biases_initalizer = tf.constant_initializer(-1.0),
			            activation_fn = tf.nn.sigmoid
			             )

		output = H*T + (1.0 - T)
		return output

###################### placeholder ##############################
input_layer = tf.placeholder(tf.float32, [None, 32,32,3], name = 'input')
label_layer = tf.placeholder(tf.int32, [ None] )
label_one_hot = slim.layers.one_hot_encoding(label_layer, 10)


###################### build the highway network #################

# firstly , the data will gou through a conv dense
output = slim.conv2d(
					input_layer, 
					63, 
					[3,3], 
					normalizer_fn = slim.batch_norm, 
					scope = 'conv_'+str(0)
					)

# and the data will get in high dense
for i in range(5):

	for j in range(number_units):
		output = highwayUnit(output, j+(i*number_units) )
	# and the data will get in a conv dense at the end of every highway unit
	top_output = slim.conv2d(
							output, 
							[3,3]
							stride =[2,2],
							normalizer_fn = slim.batch_norm,
							scope = 'conv_'+str(i))

# the top dense of network
top_output = slim.conv2d(
						output, 
						10, 
						[3,3],
						normalizer_fn = slim.batch_norm,
						activation_fn = None,
						scope = 'conv_top'
						)

# the output of network
output=   slim.layers.softmax(slim.layers.flatten(top_output))


#######################  grt loss and train ############
loss = tf.reduce_mean(-tf.reduce_sum(label_one_hot * tf.log(output) + 1e-5, reduction_indices = [1]))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)



