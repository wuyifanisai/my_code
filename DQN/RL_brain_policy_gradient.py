"""
The Dueling DQN based on this paper: https://arxiv.org/abs/1511.06581
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class Policy_Gradient(object):
	def __init__(self,
				n_actions,
				n_features,
				learning_rate=0.01,
				reward_decay = 0.95
				):
		self.n_actions = n_actions
		self.n_features = n_features
		self.learning_rate = learning_rate
		self.gamma = reward_decay

		self.ep_obs = [] # store obersvation memory in one eposide
		self.ep_as = [] #store actions memory in one eposide
		self.ep_rs = [] #store reward memory in one eposide

		self._build_net() # build network when initialization

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def _build_net(self, method='manual'):
		with tf.variable_scope("net_inputs"):
			self.tf_obs = tf.placeholder(tf.float32, 
										[None,self.n_features],
										name = "observations")
			# used as input of network

			self.tf_acts = tf.placeholder(tf.int32,[None,]) # dtype must be int for label
			# used as the label of training

			self.tf_vt = tf.placeholder(tf.float32, [None,])
			#used as the guide of train gradient
		
		def build_layers(s, n_l1, w_initializer, b_initializer, train=True):
			with tf.variable_scope('l1'):	
				w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, trainable=train)
				b1 = tf.get_variable('b_1', [1, n_l1], initializer=b_initializer,  trainable=train)
				l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer,   trainable=train)
				b2 = tf.get_variable('b_2', [1, self.n_actions], initializer=b_initializer,   trainable=train)
				out = tf.matmul(l1, w2) + b2
			return out

		all_act = build_layers(
							self.tf_obs, 
							10, 
							w_initializer = tf.random_normal_initializer(mean=0,stddev =0.3),
							b_initializer = tf.random_normal_initializer(mean=0,stddev =0.3),
							)
		
		'''
		layer = tf.layers.dense(
								inputs = self.tf_obs,
								units = 10,
								activation = tf.nn.tanh,
								kernel_initializer = tf.random_normal_initializer(mean=0,stddev =0.3),
								bias_initializer = tf.constant_initializer(0.1),
								name = 'fc_1'
								)

		all_act = tf.layers.dense(
								inputs = layer,
								units = n_actions,
								activation = None,
								kernel_initializer = tf.random_normal_initializer(mean=0,stddev=0.3),
								name = 'fc_2'
								)
								
		'''
			
		self.all_act_prob = tf.nn.softmax(all_act, name = 'act_prob')

		with tf.variable_scope('loss'):
			cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.all_act_prob, labels = self.tf_acts)
			# dtype must be int for label

			loss = tf.reduce_mean(cross_entropy*self.tf_vt)

		with tf.variable_scope('train'):
			self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

	def choose_action(self, obersvation):
		prob_weights = self.sess.run(self.all_act_prob,
									feed_dict = {self.tf_obs:obersvation[np.newaxis,:]})

		action = np.random.choice(range(prob_weights.shape[1]), p = prob_weights.ravel())
		# choose the action according to the prob
		return action

	def store_transition(self, s, a, r):
		self.ep_obs.append(s)
		self.ep_as.append(a)
		self.ep_rs.append(r)

	def learn(self):
		decay_normal_reward = self._decay_normal_reward()
		self.sess.run(self.train_op,
						feed_dict={self.tf_obs:np.vstack(self.ep_obs),
									self.tf_acts:np.array(self.ep_as),
									self.tf_vt:decay_normal_reward}
						)
		self.ep_as, self.ep_obs, self.ep_rs = [], [], []
		#clean the memory

		return decay_normal_reward
		# return for plot 

	def _decay_normal_reward(self):

		decay_eposide_reward = np.zeros_like(self.ep_rs) 

		running_step_ward = 0

		for i in reversed(range(0,len(self.ep_rs))):
			running_step_ward = running_step_ward*self.gamma + self.ep_rs[i]
			decay_eposide_reward[i] = running_step_ward
		# the reward of previous steps in a eposide is more important
		# so a decay is used to make sure the previous steps memory is more important

		decay_eposide_reward -= np.mean(decay_eposide_reward)
		decay_eposide_reward /= np.std(decay_eposide_reward)

		return decay_eposide_reward
		 




