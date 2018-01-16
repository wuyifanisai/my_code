import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


def add_layer(input_data, input_feature_num, output_feature_num,activstion=None):
	Weights = tf.Variable(tf.random_normal([input_feature_num,output_feature_num]),name = 'W')
	biases = tf.Variable(tf.random_normal([1,output_feature_num])+0.1)
	Wx_plus_b = tf.matmul(input_data,Weights) + biases
	if activstion == None:
		return Wx_plus_b
	else:
		return activstion(Wx_plus_b)
		
#定义一个用于分类问题中计算预测准确的的函数，函数引用了feed_dict
def compute_accuracy(x,y):
	global output1
	pre = sess.run(output1 , feed_dict={xs:x})
	correct_accuracy = tf.equal(tf.argmax(pre,1) , tf.argmax(y,1) )
	accuracy = tf.reduce_mean(tf.cast(correct_accuracy , tf.float32) )
	result = sess.run(accuracy , {xs:x , ys:y })
	return result


xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])

#这里没有使用隐藏层，直接连接到输出层，多分类激活输出用softmax
output1 = add_layer(xs,784,10,activstion=tf.nn.softmax)

#使用交叉熵来表征多分类问题的损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(output1) , reduction_indice=[1])) 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for step in range(1000):
	batch_xs , batch_ys = mnist.train.next_batch(100)         #使用批量处理的方式进行训练
	sess.run(train_step, feed_dict = {xs:batch_xs , ys:batch_ys})
	if step % 10 ==0:
		print('accuracy of test images ==>',compute_accuracy(mnist.test.images , mnist.test.labels))
