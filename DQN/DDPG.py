"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
tensorflow 1.0
gym 0.8.0
"""
'''
some problem ocurred when the code was running
when it was a not bad statement, the actor_net could output a nice action to keep good statement and reward
but when it was a bad statement, somtimes, the actor_net output a not good action, so it was a bit hard to transform a bad 
statement to a good one
'''


######################## import library #####################

import tensorflow as tf 
import numpy as np
import gym as gym 

################ hyper parameters ####################

MAX_EPISODES = 250  # max number of round
MAX_EP_STEPS = 200  # max number of steps in a round
LR_A = 0.001 # learning rate for actor eva net 
LR_C = 0.001 # learning rate for critic eva net
GAMMA = 0.9 # reward discount
TAU = 0.01  # soft replacement
MEMORY_SIZE = 10000
BATCH_SIZE = 32
RENDER = False
ENV_NAME = 'Pendulum-v0'
LOAD = True 
# False ==> train and save para  True==>restore the para and play

################## define class of DDPG ##################
class DDPG(object):
	def __init__(self, a_dim, s_dim, a_top,load=False):
		self.load = load
	# dimision of actions , statement, and limit of action
	
	# in the __init__ method, variable without "self" a_next, q, q_next is considered as the output from net for 
	# next action or q_target
	
	# and the variables which are placeholder are prepared for input from memory when learn 
	# tensor graph should be correct so the gradient could be right

		self.memory = np.zeros((MEMORY_SIZE, 2*s_dim + a_dim + 1),dtype = np.float32)
		# every piece of memory includes observation, observation_next, action, reward

		self.pointer = 0

		self.sess = tf.Session()

		self.a_replace_counter = 0
		self.c_replace_counter = 0

		self.a_dim = a_dim
		self.s_dim = s_dim
		self.a_top = a_top

		self.S = tf.placeholder(tf.float32, [None, s_dim], name = 's')
		# observation of this step

		self.S_ = tf.placeholder(tf.float32, [None, s_dim], name = 's_')
		# observation of next step

		self.R = tf.placeholder(tf.float32, [None, 1], name ='reward')
		# reward from action and observation

		with tf.variable_scope("actor_net"):

			# calculate action for this statement with actor eval net
			# this is a kind of eval net in DQN 
			self.a = self._bulid_actor_net(
										self.S,  # comes from memory statement
										scope = 'eval',
										trainable = True
									)

			# calculate action for next statement with actor target net
			# this is a kind of target net in DQN
			a_next = self._bulid_actor_net(
											self.S_,  # comes from memory next statement
											scope = 'target',
											trainable = False
											)

		with tf.variable_scope("critic_net"):

			# calculate Q for this statement and action with critic eval net
			# this is a kind of eval net in DQN 
			q = self._bulid_critic_net(
											self.S,   # comes from memory statement
											self.a,    # should be the action from memory !!
											scope = "eval",
											trainable = True
											)
			# calculate Q for next statement and action with critic target net
			# this is a kind of target net in DQN 
			q_ = self._bulid_critic_net(
											self.S_,   # comes from memory next statement
											a_next,   # a_next comes from actor_target according to next_statement
											scope = "target",
											trainable = False
											)

		# networks parameters
		self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_net/eval')
		self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_net/target')
		self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_net/eval')
		self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_net/target')


        # replacement for target net from eval net
		self.soft_replacement = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)] 

        # train operation for critic--------------------------------------
		q_target = self.R + GAMMA*q_
		#td_error = tf.losses.mean_squared_error(labels = q_target, prediction = self.q)
		td_error = tf.reduce_mean(tf.squared_difference(q_target, q))

		c_loss = td_error # loss of critic eval net
		self.ctrian = tf.train.AdamOptimizer(LR_C).minimize(c_loss,var_list = self.ce_params)

        # train operation for actor -------------------------------------
		a_loss = -tf.reduce_mean(q) # loss of actor eval net
		self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

		if self.load:
			saver = tf.train.Saver()
			saver.restore(self.sess ,"f://DDPG.\\DDPG.ckpt")
			print('loaded the para !')
		else:
			saver = tf.train.Saver()
			saver.restore(self.sess ,"f://DDPG.\\DDPG.ckpt")
			print('loaded the para and train again ! ! !')
			#self.sess.run(tf.global_variables_initializer())
        # when __init__ of class DDPG is executed, all the variable were defined, 
        # and it is time to run initializer()

	def choose_action(self, s):
    	# choose action for this statement
		return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

	def replacement(self):
		# soft parameters replacement for target net 
		self.sess.run(self.soft_replacement)

	def learn(self):
 
    	# choose some piece of memory from memory hub
		indices = np.random.choice(MEMORY_SIZE, size = BATCH_SIZE) 

		bt = self.memory[indices,:] #some memory picked
		bs = bt[: , :self.s_dim] #statement of bt
		ba = bt[: , self.s_dim:self.s_dim + self.a_dim] # action of bt
		br = bt[:, -self.s_dim-1 : -self.s_dim] #reward of _bulid_actor_net
		bs_ = bt[: , -self.s_dim :] #next statement of bt

    	# learn for actor_eval_net-------------------------------------------
    	# by running atrain,we need a_loss <== self.q <== bs and ba 	
		self.sess.run(self.atrain, feed_dict = {self.S:bs, self.a: ba})

    	# learn for critic_eval_net------------------------------------------------
    	# by running ctrain,we need c_loss <== self.q and self.q_ and self.R 
    	# <== bs, ba, bs_, br, self.a_  
		self.sess.run(self.ctrian, feed_dict = {self.R: br, self.S:bs, self.S_:bs_, self.a:ba})


	def store_memory(self, s, a, r, s_):
		m = np.hstack((s, a, [r], s_))
		index = self.pointer%MEMORY_SIZE
		self.memory[index,:] = m
		self.pointer += 1

	def _bulid_actor_net(self, s, scope, trainable):
		with tf.variable_scope(scope):
			w_initializer = tf.random_normal_initializer(mean=0,stddev =0.3)
			b_initializer = tf.random_normal_initializer(mean=0,stddev =0.3)
			n_l1 = 10
			with tf.variable_scope('l1'):	
				w1 = tf.get_variable('w1', [self.s_dim, n_l1], initializer=w_initializer, trainable=trainable)
				b1 = tf.get_variable('b_1', [1, n_l1], initializer=b_initializer,  trainable=trainable)
				l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2', [n_l1, self.a_dim], initializer=w_initializer,   trainable=trainable)
				b2 = tf.get_variable('b_2', [1, self.a_dim], initializer=b_initializer,   trainable=trainable)
				action = tf.nn.tanh(tf.matmul(l1, w2) + b2)
			return tf.multiply(action, self.a_top*1.2, name = "scaled_a")
			

	def _bulid_critic_net(self, s, a, scope, trainable):
		with tf.variable_scope(scope):
			n_l1 = 10
			w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
			w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
			b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
			net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

			w_initializer = tf.random_normal_initializer(mean=0,stddev =0.3)
			b_initializer = tf.random_normal_initializer(mean=0,stddev =0.3)

			w2 = tf.get_variable('w1', [n_l1, 1], initializer=w_initializer, trainable=trainable)
			b2 = tf.get_variable('b_1', [1, 1], initializer=b_initializer,  trainable=trainable)
			q_value = tf.matmul(net, w2) + b2
			return q_value  # Q(s,a)

	def save_para(self):
		# store -----
		saver = tf.train.Saver()
		save_path = saver.save(self.sess, "f://DDPG_//DDPG.ckpt")
		print('save to path:' , save_path)




######################## RL training #######################################
env = gym.make(ENV_NAME)
env = env.unwrapped
#env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
print(a_bound)

ddpg = DDPG(a_dim, s_dim, a_bound,load = LOAD)  #load = True means use the para saved befoe

var=1
for i in range(MAX_EPISODES):
	print("round-->",i)
	s = env.reset()
	ep_reward = 0
	for j in range(MAX_EP_STEPS):
		print("step-->",j)

		if LOAD:  # just play , do not learn
			env.render()

			a = ddpg.choose_action(s)
			#a = [np.clip(np.random.normal(a, var), -2, 2)] 
			# add randomness to action selection for exploration
			print(a)

			s_, r, done, info = env.step(a)
			s = s_
			ep_reward = r
			print('Episode:', i, ' Reward: %i' % int(ep_reward))


		else:  # play and learn 

			if RENDER:
				env.render()

			a = ddpg.choose_action(s)

			a = [np.clip(np.random.normal(a, var), -2, 2)] 
			# add randomness to action selection for exploration
			print(a)

			s_, r, done, info = env.step(a)
			#print(s, s_)
		
			if r>-3:
				r=r+3
		
			ddpg.store_memory(s, a, r/10, s_)

			if ddpg.pointer > MEMORY_SIZE:
				var *= .9995
			
				ddpg.learn()
				print("DDPG learn !")
				ddpg.replacement()
				print("replacement!")

			s = s_
			ep_reward = r

		
			print('Episode:', i, ' Reward: %i' % int(ep_reward))
			#print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var)
			if i > MAX_EPISODES*0.8:RENDER = True
			if j == MAX_EP_STEPS-1:
				break

ddpg.save_para()


















