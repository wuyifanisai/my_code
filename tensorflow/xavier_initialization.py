# use xavier_initializer in tensorflow
import tensorflow as tf
a = tf.get_variable("a", shape=[4, 4], initializer=tf.contrib.layers.xavier_initializer())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))