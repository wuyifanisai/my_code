
	import tensorflow as tf


	from tensorflow.examples.tutorials.mnist import input_data


	mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


	



	#定义一个用于分类问题中计算预测准确的的函数，函数引用了feed_dict


	def compute_accuracy(x,y):


	    global output_dense4


	    pre = sess.run(output1 , feed_dict={xs:x})


	    correct_accuracy = tf.equal(tf.argmax(pre,1) , tf.argmax(y,1) )


	    accuracy = tf.reduce_mean(tf.cast(correct_accuracy , tf.float32) )


	    result = sess.run(accuracy , {xs:x , ys:y ,keep_prob : 0.6})


	    return result


	



	def weight_variable(shape):


	    initial = tf.truncated_normal(shape, stddev=0.1,)


	    return tf.Variable(inital)


	def biase_variable(shape):


	    initial = tf.constant(0.1,shape = shape)


	    return tf.Variable(initial)


	def conv2d(x,W):


	    return tf.nn.conv2d(x,W,strides=[1,1,1,1] , padding= 'SAME')


	def max_pooling(x):


	    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=2,padding='SAME')


	



	#define the placeholder ---------------------


	xs = tf.placeholder(tf.float32 , [None,784])


	ys = tf.placeholder(tf.float32 , [None,10])


	keep_prob = tf.placeholder(tf.float32)


	x_image = tf.reshape(xs,[-1,28,28,1])


	



	



	# conv1 layer ---------------------------------


	W_conv1 = weight_variable([5,5,1,32])


	b_conv1 = biase_variable([32])


	output_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)


	#output_size : -1*28*28*32


	output_conv1 = max_pooling(output_conv1)


	#output_size : -1*14*14*32


	



	# conv2 layer -----------------------------------


	W_conv2 = weight_variable([5,5,32,64])


	b_conv2 = biase_variable([64])


	output_conv2 = tf.nn.relu(conv2d(output_conv2 , W_conv2)+b_conv2)


	#output_size : -1*14*14*64


	output_conv2 = max_pooling(output_conv2)


	#output_size : -1*7*7*64


	



	# Dense 3 layer--------------------------------- 


	W_dense3 = weight_variable([7*7*64,1024])


	b_dense3 = biase_variable([1024])


	output_conv2_fatten = tf.reshape(output_conv2 , [-1,7*7*64])


	output_dense3 =tf.nn.relu(tf.matmul(output_conv2_fatten , W_dense3) + b_dense3)


	output_dense3 = tf.nn.dropout(output_dense3 , keep_prob)


	



	# Dense4 layer ---------------------------


	W_dense4 = weight_variable([1024,10])


	b_dense4 = biase_variable([10])


	output_dense4 =tf.nn.softmax(tf.matmul(output_dense3 , W_dense4) + b_dense4)


	pre =output_dense4


	



	#使用交叉熵来表征多分类问题的损失------


	cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(output_dense4) , reduction_indices=[1]))


	train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)


	



	sess = tf.Session()


	sess.run(tf.initialize_all_variables())


	



	for step in range(1000):


	    batch_xs , batch_ys = mnist.train.next_batch(100)         #使用批量处理的方式进行训练


	    sess.run(train_step, feed_dict = {xs:batch_xs , ys:batch_ys,keep_prob:0.6})


	    if step % 10 ==0:


	        print('accuracy of test images ==>',compute_accuracy(mnist.test.images , mnist.test.labels))


	



	



	



	

