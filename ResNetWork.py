'''
ResNet in tf
for mnist classification
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# use the minst data as the data for res_net

##################### read the minst data ########################
print("read minst data ...")
minst = input_data.read_data_sets("dir/" , one_hot = True)
# label ==> one_hot

##################### build some function for building resnet ####################

IS_TRAINING = True # for bn_operation

def get_tf_var(shape, name):
	# give you the variables in the shape you want
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1), name = name)

def full_dense(x,shape):
	# layer of full-connection
	w = tf.Variable(shape) #shape contains the dimension of input and output
	b = tf.Variable(tf.zeros([shape[-1]]))
	return tf.matmul(x, w) + b

def batch_norm(input, output_conv_form = True, decay = 0.99):
	# batch norm operation

	scale = tf.Variable(tf.ones([ input.get_shape()[-1] ]))
	# scale operation in bn 

	beta = tf.Variable(tf.zeros([ input.get_shape()[-1] ]))
	# shift operation in bn

	mean = tf.Variable(tf.zeros([input.get_shape()[-1]]), trainable = False )
	var = tf.Variable(tf.ones(input.get_shape([-1])),trainable = False )
	# obviously , mean and var are not trainable ! they are from data itself

	if IS_TRAINING:
		# if it is during training, mean and var should be caculated from data

		if output_conv_form:
			# if output is a kind of conv_form, mean and var should be caculated form
			# all dimension

			batch_mean , batch_var = tf.nn.moments(input, axes = [0,1,2] )
		else:
			batch_mean , batch_var = tf.nn.moments(input, axes = [0] )

		# when we get the mean and vart, let us do bn operation
		train_mean_op = tf.assign(mean, mean*decay + (1 - decay)*batch_mean)
		train_var_op = tf.assign(var, var*decay + (1 - decay)*batch_var)

		with tf.dependcies([train_mean_op , train_var_op]):
			return tf.nn.bact_normlization(
											input,
											batch_mean,
											batch_var,
											beta,
											scale,
											0.001)
	# if this is not training
	else:
		return tf.nn.bact_normlization(input, mean, var, beta, scale, 0.001)
		# this maybe not correct

def conv_with_batch_norm_2d(x, filter_shape, stride):
	filter_window = get_tf_var(filter_shape)
	conv_output = tf.nn.conv2d(
								x, 
								filter = filter_window,
								strides = [1, stride ,stride, 1],
								padding = 'SAME')

	bn_output = batch_norm(conv_output)

	return tf.nn.relu(bn_output)

def conv_no_batch_norm_2d(x, filter_shape, stride):
	out_channels = filter_shape[-3] 
	# there are 4 dimension in filter

	conv = tf.nn.conv2d(
						x, 
						filter = get_tf_var(filter_shape),
						strides = [1, stride, stride ,1],
						padding = 'SAME'
						)
	bias = tf.Variable(tf.zeros([out_channels]), name = 'bias') 

	return tf.nn.relu(tf.nn.bias_add(conv , bias))














