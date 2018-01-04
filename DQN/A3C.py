'''
A3C network for RL learning
environment of The Pendulum example.
''' 

import tensorflow as tf
import numpy as np 
import gym
import matplotlib.pyplot as plt 

import multiprocessing
import threading 
import os 
import shutil

############ global variable ###################

# for thr net
N_WORKERS = multiprocessing.cpu_count()
MAX_EP_STEP = 200 # most steps in a ep
MAX_GLOBAL_EP = 2000 # most ep in a train_op
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10  # update global net every 10 tiers 
GAMMA = 0.9 # hyperparamters of Q-learning
ENTROPY_BETA = 0.01 
LR_A = 0.0001 # learning rate for actor
LR_C = 0.001 # learning rate for critic
GLOBAL_RUNNING_R = []  # running reward
GLOBAL_EP = 0 

# for the envrionment
GAME = 'Pendulum-v0'
env = gym.make(GAME)
N_S = env.observation_space.shape[0] # dimension of statement
N_A = env.action_space.shape[0] #dimension of action
A_BOUND = [env.action_space.low, env.action_space.high] # bound of action


############# define ACNet ###################
class ACNet(object):
	def __init__(self, scope, globalAC):

		# in the __init__ , we defined some important variables to make some function
		# such as define loss, train_op
		# define some important tensor graph ...

		if scope == GLOBAL_NET_SCOPE: # let us make a global net

			with tf.variable_scope(scope):

				# give me some placeholders, come on !
				self.s = tf.placeholder(tf.float32, [None, N_S],'S')

				# the network will return para according to self.s
				# para of action net and critic net
				self.a_para, self.c_para = self._build_net(scope)[-2:]

		else: # let us make a local worker network
			with tf.variable_scope(scope):

				# give me some placeholder to give the net

				# this is the input of net
				self.s = self.s = tf.placeholder(tf.float32, [None, N_S],'S')

				# this is the action from memory
				self.a_memory = tf.placeholder(tf.float32, [None, A_S],'A')

				# this is the value target of q_value
				self.v_target = tf.placeholder(tf.float32, [None, 1],'v_target')

				# the network will return para according to self.s
				# para of action net and critic net
				# mu and sigma are the output about chosen action from actio_net
				# mu and sigma are the parameters of a normal distribution
				# self.v is the value of this statement
				mu, sigma, self.v, self.a_para, self.c_para = self._build_net(scope)

				# we need self,v_target and self.v to grt c_loss 
				td = tf.subtract(self.v_target, self.v, name ='td_error')
				# this is the the loss for q_learning , for the train_operation of critic_net

				with tf.variable_scope('c_loss'):
					self.c_loss = tf.reduce_mean(tf.squared(td))


				with tf.variable_scope('get_action_distribution'):
					mu = mu*A_BOUND[1]
					sigma += 1e-4
					normal_dist = tf.distributions.Normal(mu, sigma)


				with tf.variable_scope('a_loss'):
					# we need the action from memory to get a_loss
					log_prob = normal.dist.log_prob(self.a_memory)

					error = log_prob*td

					entropy = normal_dist.entropy() # encourage exploration

					error = ENTROPY_BETA * entropy + error

					self.a_loss = tf.reduce_mean(error)

				with tf.variable_scope('chosen_action'):
					# use the action_net of local net to choose action
					self.a = tf.clip_by_value(
											tf.squeeze(
														normal_dist.sample(1),
														axis = 0
														),
											A_BOUND[0],
											A_BOUND[1]
											)

				with tf.variable_scope('local_gradient'):
					# get the gradient of local net
					# to train local network and update global network
					self.a_grad = tf.gradient(self.a_loss, self.a_para)
					self.c_grad = tf.gradient(self.c_loss, self.c_para)

				with tf.variable_scope('sync'):
					# todo


				with tf.variable_scope('pull'):
					# pull the para of global action_net to the local action_net
					self.pull_a_para_op = [local_para.assign(global_para) for local_para, global_para in zip(self.a_para, globalAC.a_para)]

					# pull the para of global critic_net to the local critic_net
					self.pull_c_para_op = [local_para.assign(global_para) for local_para, global_para in zip(self.c_para, globalAC.c_para)]


				with tf.variable_scope('push'):
					# push the gradients of training to the global net
					# use the gradients caculated from local net to train global net

					self.update_gradient_action_op = optimizer_action.apply_gradients(zip(self.a_grad, globalAC.a_para))
					self.update_gradient_critic_op = optimizer_critic.apply_gradients(zip(self.c_para, globalAC.c_para))



	def _build_net(self, scope):
		# to define a network structure for action_net ,critic_net in global and local network
		w_init = tf.random_normal_initializer(0.0, 0.1)

		with tf.variable_scope('actor'):
			# we will get some normal_distributions of action, number of distributions is N_A
			output_a = tf.layers.dense(
										self.s,
										20,
										tf.nn.relu6,
										kernel_initializer = w_init,
										name = 'output_a'
										)

			mu = tf.layers.dense(  # get the mu of a normal distribution of action, dim of mu is N_A
									output_a,
									N_A,
									tf.nn.tanh,
									kernel_initializer = w_init,
									name = 'mu'
								)

			sigma = tf.layers.dense( # get the sigma of a normal distribution of action, dim of sigma is N_A
									output_a,
									N_A,
									tf.nn.softplus,
									kernel_initializer = w_init,
									name = 'sigma'
								)

		with tf.variable_scope('critic'):
			output_c = tf.layers.dense(
										self.s,
										20,
										tf.nn.relu6,
										kernel_initializer = w_init,
										name = 'output_c'
										)

			v = tf.layers.dense(  # we get the value of this statement self.s
								output_c,
								1,
								kernel_initializer = w_init,
								name = 'v'
								)

		a_para = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope+'/actor')
		c_para = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope+'/critic')

		return mu, sigma, v, a_para, c_para

 
	def update_global(self, feed_dict): # push the gradients to the global net to train
		# to train global net using the gradiients caculated from local net

		SESS.run([self.update_gradient_action_op, self.update_gradient_critic_op], feed_dict)
		# some data is from placeholder

	def pull_global(self): #pull the new para from global net to local net
		SESS.run([self.pull_a_para_op, self.pull_c_para_op])

	def choose_action(self, s):
		# we need the statement of this moment to caculate a action
		s = s[np.new.axis, :]

		return SESS.run(self.a, {self.s:s})[0]
		# we need figure out the structure of output action

class Worker(object):
	""" define local worker net , and train with envrionment itself """
	def __init__(self, name, globalAC):
		
		self.env = gym.make(GAME).unwarpped

		self.name = name

		self.AC = ACNet(name,globalAC)
		# globalAC is the global net connecting with the local net

	def work(self):
		global GLOBAL_RUNNNING_R, GOBAL_EP

		total_step = 1
		buffer_s = []
		buffer_a = []
		buffer_r = []

		# let us train 
		while not COORP.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
			s = self.env.reset()

			ep_r = 0

			# let us do every step
			for i in range(MAX_EP_STEP):

				# get the action for now
				a = self.ACNet.choose_action(s)

				# interaction with envrionment
				s_, r, done, info = self.env.step(a)

				done = True if i ==MAX_EP_STEP else False

				ep_r += r

				# store
				buffer_s.append(s)
				buffer_a.append(a)
				buffer_r.append((r+8)/8)

				# shuold update global net ?------------------------------------
				if total_step%UPDATE_GLOBAL_ITER ==0 or done:
					if done:
						v_s_ = 0 
						# it is the value of next statement
						# if it is done, we assume it is 0

					else:
						v_s_ = SESS.run(self.AC.v, {self.AC.s:s_[np.newaxis, :]})[0, 0]

					buffer_v_target = []
					#store the v_target for caculating the loss

					for r in buffer_r[::-1]: # reverse the buffer_r
						v_target_element = r + GAMMA*v_s_
						buffer_v_target.append(v_target_element)

					buffer_v_target.reverse()

					buffer_s = np.vstack(buffer_s)
					buffer_a = np.vstack(buffer_a)
					buffer_v_target = np.vstack(buffer_v_target)

					# update global_net para with gradients from local net 
					# and give the new para from global net to local net
					feed_dict = {
									self.AC.s: buffer_s,
									self.AC.a_memory: a,
									self.AC.v_target: buffer_v_target
								}

					# here are caculating gradients from local net
					# and push them to global net to update para of global net
					self.AC.update_global(feed_dict)

					# and givr the new para to local net
					self.AC.pull_global()

					buffer_s = []
					buffer_a = []
					buffer_r = []



				s = s_
				total_step += 1

				if done:
					# put running reward of every step into GLOBAL_RUNNING_R
					if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        			GLOBAL_RUNNING_R.append(ep_r)
                    			else:
                       		 		GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    			print(
                        			self.name,
                        			"Ep:", GLOBAL_EP,
                        			"| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          			)
                   			GLOBAL_EP += 1
                    			break

























		
















































