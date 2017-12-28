

'''
dense deep networks
'''
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

total_layers = 50
number_units = total_layers/5


####################### define highway unit ########################
def denseBlock(input_layer, i, j):
	with tf.variable('dense_net'+str(i)):
		nodes = []
		a = slim.conv2d(
						input_layer, 
						64, [3,3], 
						normalizer_fn =slim.batch_norm
						)
		nodes.append(a)

		for _ in range(j):
			b = slim.conv2d(
							tf.concat(3,nodes), 
							#3 means concat operation for dim in channels 
							64, 
							[3.3],
			            	normalizer_fn =slim.batch_norm
			             	)
			nodes.append(b)
		return b

###################### placeholder ##############################
input_layer = tf.placeholder(tf.float32, [None, 32,32,3], name = 'input')
label_layer = tf.placeholder(tf.int32, [ None] )
label_one_hot = slim.layers.one_hot_encoding(label_layer, 10)


###################### build the dense network #################

# firstly , the data will gou through a conv dense
output = slim.conv2d(
					input_layer, 
					63, 
					[3,3], 
					normalizer_fn = slim.batch_norm, 
					scope = 'conv_'+str(0)
					)

# and , the data will get in dense units
for i in range(5): # 5 denseBlocks
	output = denseBlock(output , i,number_units)
	output = slim.conv2d(
						output, 
						64,
						[3,3],
						stride =[2,2],
						normalizer_fn = slim.batch_norm,
						scope = 'conv_i'
						)

# and ,the data will get in final conv dense
top_output = slim.conv2d(
						output, 
						10,
						[3,3],
						normalizer_fn =slim.batch_norm,
						activation_fn = None,
						scope = 'conv_top'
						)

output = slim.layers.softmax(slim.layers.flatten(top_output))

#######################  grt loss and train ############
loss = tf.reduce_mean(-tf.reduce_sum(label_one_hot * tf.log(output) + 1e-5, reduction_indices = [1]))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)



