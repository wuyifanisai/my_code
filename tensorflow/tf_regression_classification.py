

	import tensorflow as tf


	import matplotlib.pyplot as plt


	import numpy as np


	



	tf.set_random_seed(1)


	np.random.seed(1)


	



	# fake data


	x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)


	noise = np.random.normal(0, 0.1, size=x.shape)


	y = np.power(x, 2) + noise                          # shape (100, 1) + some noise


	# plot data


	plt.scatter(x, y)


	plt.show()


	



	tf_x = tf.placeholder(tf.float32, x.shape)     # input x


	tf_y = tf.placeholder(tf.float32, y.shape)     # input y


	



	# neural network layers


	l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)          # hidden layer


	output = tf.layers.dense(l1, 1)                     # output layer


	loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost


	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)


	train_op = optimizer.minimize(loss)


	sess = tf.Session()                                 # control training and others


	sess.run(tf.global_variables_initializer())         # initialize var in graph


	



	plt.ion()   # something about plotting


	



	for step in range(100):


	    # train and net output


	    _, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})


	    if step % 5 == 0:


	        # plot and show learning process


	        plt.cla()


	        plt.scatter(x, y)


	        plt.plot(x, pred, 'r-', lw=5)


	        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})


	        plt.pause(0.1)


	plt.ioff()


	plt.show()


	



	



	#two features input####--------------------


	import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

from  mpl_toolkits.mplot3d import Axes3D



# prepare data -----------------

# for train 

X= np.linspace(-1,1,200)

np.random.shuffle(X)

X=X.reshape(([100,2]))

Y = np.power(X[:,0],2) + np.power(X[:,1],3) 

noise = np.random.normal(0,0.1,Y.shape)

Y = Y + noise

Y = Y.reshape(([-1,1]))



#for test

X1= np.linspace(-1,1,200)

np.random.shuffle(X1)

X1=X1.reshape(([100,2]))

Y1 = np.power(X1[:,0],2) + np.power(X1[:,1],3) 

noise = np.random.normal(0,0.1,Y1.shape)

Y1 = Y1 + noise

Y1 = Y1.reshape(([-1,1]))





def add_layer(input_data, input_feature_num, output_feature_num,activstion=None):

    Weights = tf.Variable(tf.random_normal([input_feature_num,output_feature_num]))

    biases = tf.Variable(tf.random_normal([1,output_feature_num])+0.1)

    Wx_plus_b = tf.matmul(input_data,Weights) + biases

    if activstion == None:

        return Wx_plus_b

    else:

        return activstion(Wx_plus_b)



tf_x = tf.placeholder(tf.float32, X.shape)

tf_y = tf.placeholder(tf.float32, Y.shape)



# bulid nerual network with tf-----------------------

output_1 = add_layer(tf_x,2,10,tf.nn.relu)

output_2 = add_layer(output_1,10,1,)

#output_3 = add_layer(output_2,3,1)



loss = tf.losses.mean_squared_error(tf_y , output_2)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

train_op = optimizer.minimize(loss)



# session running ---------------------

sess =tf.Session()

sess.run(tf.global_variables_initializer())

cost_list=[]

for step in range(20000):

    _,cost,pred = sess.run([ train_op, loss , output_2],{tf_x:X ,tf_y:Y })

    if step % 10==0 and step > 1000:

        cost_list.append(cost)

        print(step , cost)

plt.plot(cost_list)

plt.show()

plt.scatter(Y,pred,c='r')

plt.show()



pred1 = sess.run(output_2, {tf_x:X1})

plt.scatter(Y1,pred1,c='r')

plt.show()

print(sess.run(tf.losses.mean_squared_error(tf_y , pred1),{tf_y:Y1 }))
