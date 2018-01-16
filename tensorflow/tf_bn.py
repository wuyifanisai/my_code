import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
exit()
# neural network with 5 layers
#
# · · · · · · · · · ·          (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/       -- fully connected layer (sigmoid+BN)   W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/         -- fully connected layer (sigmoid+BN)   W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/           -- fully connected layer (sigmoid+BN)   W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/             -- fully connected layer (sigmoid+BN)   W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]——
#         \x/               -- fully connected layer (softmax+BN)   W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# variable learning rate
lr = tf.placeholder(tf.float32)
# train/test selector for batch normalisation
tst = tf.placeholder(tf.bool)
# training iteration
iter = tf.placeholder(tf.int32)

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 200
M = 100
N = 60
P = 30
Q = 10

# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1),name = 'W1')  # 784 = 28 * 28
S1 = tf.Variable(tf.ones([L]),name = 'S1')
O1 = tf.Variable(tf.zeros([L]),name = 'O1')
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1),name = 'W2')
S2 = tf.Variable(tf.ones([M]),name = 'S2')
O2 = tf.Variable(tf.zeros([M]),name = 'O2')
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1),name = 'W3')
S3 = tf.Variable(tf.ones([N]),name = 'S3')
O3 = tf.Variable(tf.zeros([N]),name = 'O3')
W4 = tf.Variable(tf.truncated_normal([N, P], stddev=0.1),name = 'W4')
S4 = tf.Variable(tf.ones([P]),name = 'S4')
O4 = tf.Variable(tf.zeros([P]),name = 'O4')
W5 = tf.Variable(tf.truncated_normal([P, Q], stddev=0.1),name = 'W5')
B5 = tf.Variable(tf.zeros([Q]),name = 'B5')


## Batch normalisation conclusions with sigmoid activation function:
# BN is applied between logits and the activation function
# On Sigmoids it is very clear that without BN, the sigmoids saturate, with BN, they output
# a clean gaussian distribution of values, especially with high initial learning rates.

# sigmoid, no batch-norm, lr(0.003, 0.0001, 2000) => 97.5%
# sigmoid, batch-norm lr(0.03, 0.0001, 1000) => 98%
# sigmoid, batch-norm, no offsets => 97.3%
# sigmoid, batch-norm, no scales => 98.1% but cannot hold fast learning rate at start
# sigmoid, batch-norm, no scales, no offsets => 96%

# Both scales and offsets are useful with sigmoids.
# With RELUs, the scale variables can be omitted.
# Biases are not useful with batch norm, offsets are to be used instead

# Steady 98.5% accuracy using these parameters:
# moving average decay: 0.998 (equivalent to averaging over two epochs)
# learning rate decay from 0.03 to 0.0001 speed 1000 => max 98.59 at 6500 iterations, 98.54 at 10K it,  98% at 1300it, 98.5% at 3200it

def batchnorm(Ylogits, Offset, Scale, is_test, iteration):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.998, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, Offset, Scale, bnepsilon)
    return Ybn, update_moving_averages, exp_moving_avg.average(mean), exp_moving_avg.average(variance)  #return the shadow variable of mean, variance

def no_batchnorm(Ylogits, Offset, Scale, is_test, iteration):
    return Ylogits, tf.no_op()

# The model building --------------------------
XX = tf.reshape(X, [-1, 784])

Y1l = tf.matmul(XX, W1)
Y1bn, update_ema1, shadow_mean1 , shadow_var1 = batchnorm(Y1l, O1, S1, tst, iter)
Y1 = tf.nn.sigmoid(Y1bn)

Y2l = tf.matmul(Y1, W2)
Y2bn, update_ema2, shadow_mean2 , shadow_var2 = batchnorm(Y2l, O2, S2, tst, iter)
Y2 = tf.nn.sigmoid(Y2bn)

Y3l = tf.matmul(Y2, W3)
Y3bn, update_ema3, shadow_mean3 , shadow_var3 = batchnorm(Y3l, O3, S3, tst, iter)
Y3 = tf.nn.sigmoid(Y3bn)

Y4l = tf.matmul(Y3, W4)
Y4bn, update_ema4, shadow_mean4 , shadow_var4 = batchnorm(Y4l, O4, S4, tst, iter)
Y4 = tf.nn.sigmoid(Y4bn)

Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# collection of ema operation and shadow variable------
update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)
mean_ema_average = tf.group(shadow_mean1, shadow_mean2, shadow_mean3, shadow_mean4)
var_ema_average = tf.group(shadow_var1, shadow_var2, shadow_var3, shadow_var4)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# create new variables to store shadow variables 
mean1 = tf.Variable(tf.truncated_normal(shadow_mean1.get_shape(), stddev=0.1),name = 'M1')
mean1_op = tf.assign(mean1 , shadow_mean1)
var1 = tf.Variable(tf.truncated_normal(shadow_var1.get_shape(), stddev=0.1),name = 'V1')
var1_op = tf.assign(var1 , shadow_var1)

mean2 = tf.Variable(tf.truncated_normal(shadow_mean2.get_shape(), stddev=0.1),name = 'M2')
mean2_op = tf.assign(mean2 , shadow_mean2)
var2 = tf.Variable(tf.truncated_normal(shadow_var2.get_shape(), stddev=0.1),name = 'V2')
var2_op = tf.assign(var2 , shadow_var2)

mean3 = tf.Variable(tf.truncated_normal(shadow_mean3.get_shape(), stddev=0.1),name = 'M3')
mean3_op = tf.assign(mean3 , shadow_mean3)
var3 = tf.Variable(tf.truncated_normal(shadow_var1.get_shape(), stddev=0.1),name = 'V3')
var3_op = tf.assign(var3 , shadow_var3)

mean4 = tf.Variable(tf.truncated_normal(shadow_mean4.get_shape(), stddev=0.1),name = 'M4')
mean4_op = tf.assign(mean4 , shadow_mean4)
var4 = tf.Variable(tf.truncated_normal(shadow_var4.get_shape(), stddev=0.1),name = 'V4')
var4_op = tf.assign(var4, shadow_var4)


#=================================  Session  =======================================
# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)                           

# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, is_test, is_train):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # learning rate decay (with batch norm)
    max_learning_rate = 0.03
    min_learning_rate = 0.0001
    decay_speed = 1000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    # compute training values for output during the training
    if is_train:
        a, c = sess.run([accuracy, cross_entropy], {X: batch_X, Y_: batch_Y, tst: False})  # 'tst' is the is_test for batch_nomal function
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")

    # compute test values for output during the training
    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], {X: mnist.test.images, Y_: mnist.test.labels, tst: True})
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
     

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False})

    sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i})
    #sess.run(mean_ema_average,{X: batch_X, Y_: batch_Y, tst: False, iter: i})
    #sess.run(var_ema_average,{X: batch_X, Y_: batch_Y, tst: False, iter: i})

   
# DO TRAIN !!!!!
for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)
# save the final shadow variable to new variables 
sess.run(tf.group(mean1_op, mean2_op, mean3_op, mean4_op),{X: batch_X, Y_: batch_Y, tst: False, iter: i})
sess.run(tf.group(var1_op, var2_op, var3_op, var4_op),{X: batch_X, Y_: batch_Y, tst: False, iter: i})

# save the variable ----------------------------------
# varibale to save is below:
'''
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
S1 = tf.Variable(tf.ones([L]))
O1 = tf.Variable(tf.zeros([L]))
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
S2 = tf.Variable(tf.ones([M]))
O2 = tf.Variable(tf.zeros([M]))
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
S3 = tf.Variable(tf.ones([N]))
O3 = tf.Variable(tf.zeros([N]))
W4 = tf.Variable(tf.truncated_normal([N, P], stddev=0.1))
S4 = tf.Variable(tf.ones([P]))
O4 = tf.Variable(tf.zeros([P]))
W5 = tf.Variable(tf.truncated_normal([P, Q], stddev=0.1))
B5 = tf.Variable(tf.zeros([Q]))

mean1
var1
mean2
var2 
mean3
var3
mean4
var4

'''
saver = tf.train.Saver()
save_path = saver.save(sess, "E://nn_bn_save_net.ckpt")
print('save to path:' , save_path)
sess.close()


###################################### REBULID THE MODEL AND RESTORE PARAMETERS TO DO PREDICTION ############################
# redifine the variables to restore values from saved file
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1),name = 'W1')  # 784 = 28 * 28
S1 = tf.Variable(tf.ones([L]),name = 'S1')
O1 = tf.Variable(tf.zeros([L]),name = 'O1')
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1),name = 'W2')
S2 = tf.Variable(tf.ones([M]),name = 'S2')
O2 = tf.Variable(tf.zeros([M]),name = 'O2')
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1),name = 'W3')
S3 = tf.Variable(tf.ones([N]),name = 'S3')
O3 = tf.Variable(tf.zeros([N]),name = 'O3')
W4 = tf.Variable(tf.truncated_normal([N, P], stddev=0.1),name = 'W4')
S4 = tf.Variable(tf.ones([P]),name = 'S4')
O4 = tf.Variable(tf.zeros([P]),name = 'O4')
W5 = tf.Variable(tf.truncated_normal([P, Q], stddev=0.1),name = 'W5')
B5 = tf.Variable(tf.zeros([Q]),name = 'B5')

mean1 = tf.Variable(tf.truncated_normal(shadow_mean1.get_shape(), stddev=0.1),name = 'M1')
var1 = tf.Variable(tf.truncated_normal(shadow_var1.get_shape(), stddev=0.1),name = 'V1')

mean2 = tf.Variable(tf.truncated_normal(shadow_mean2.get_shape(), stddev=0.1),name = 'M2')
var2 = tf.Variable(tf.truncated_normal(shadow_var2.get_shape(), stddev=0.1),name = 'V2')

mean3 = tf.Variable(tf.truncated_normal(shadow_mean3.get_shape(), stddev=0.1),name = 'M3')
var3 = tf.Variable(tf.truncated_normal(shadow_var1.get_shape(), stddev=0.1),name = 'V3')

mean4 = tf.Variable(tf.truncated_normal(shadow_mean4.get_shape(), stddev=0.1),name = 'M4')
var4 = tf.Variable(tf.truncated_normal(shadow_var4.get_shape(), stddev=0.1),name = 'V4')

# restore the variables
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess , "E://./save_net.ckpt" )

# bulid the Forward propagation to predict-----------------
def Forward_propagation(X):   # input X shape is : [None, 28, 28, 1]
	XX = tf.reshape(X, [-1, 784])
	
	Y1l = tf.matmul(XX, W1)
	Y1bn = tf.nn.batch_normalization(Y1l, mean1, var1, O1, S1, 0.001)
	Y1 = tf.nn.sigmoid(Y1bn)

	Y2l = tf.matmul(Y1, W2)
	Y2bn = tf.nn.batch_normalization(y2l, mean2, var2, O2, S2, 0.001)
	Y2 = tf.nn.sigmoid(Y2bn)

	Y3l = tf.matmul(Y2, W3)
	Y3bn = tf.nn.batch_normalization(y3l, mean3, var3, O3, S3, 0.001)
	Y3 = tf.nn.sigmoid(Y3bn)

	Y4l = tf.matmul(Y3, W4)
	Y4bn = tf.nn.batch_normalization(y4l, mean4, var4, O4, S4, 0.001)
	Y4 = tf.nn.sigmoid(Y4bn)

	Ylogits = tf.matmul(Y4, W5) + B5
	Y = tf.nn.softmax(Ylogits)
	return Y


X = mnist.test.images  #[None, 28, 28, 1]
Y = Forward_propagation(X)
Y_= mnist.test.labels  #[None,10]

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('accuracy of test PREDICTION is:',accuracy)
