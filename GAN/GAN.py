'''
这是一个简单的GAN例子，其中生成器目标是生成一段等差数列，判别器目标是能够尊却判别一段数列是否为等差数列。
G网路输入为批量的随机数列
D网络输入为批量的G网络生成的数列以及批量的等差数列样本
G网络采用DNN或者RNN实现，代码中优先使用DNN
D网络采用RNN实现

代码实现方式：
定义一个类：GAN_for_timeseq
类中实现了GAN的两个G，D两个网络结构，以及gan的train函数
gan = GAN_for_timeseq(len_seq , lr_d = 0.001 , lr_g = 0.01 )
gan.train_op(batch_size , num_iter , ground_truth_tensor , None , None)

实现中的注意点：
1.SCOPE变量命名规范，方便定义不同网络优化函数的时候选择相对应的trainable变量
例如： tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'G_net')

2.由于D网络的输入包括两个部分，所以在创建graph的时候 ， 会调用两次D网络的搭建函数，因为在搭建函数中采用了scope命名的方式，所以会导致变量重复定义的情况，如下所示
# the output of d_network when input is ground_truth_tensor
self.d_output_ground_truth = self.Discriminator(self.ground_truth_tensor,reuse=False)
self.d_output_fake = self.Discriminator(self.g_output , reuse=True)
因为都是在同一个scope中，两次调用就会定义变量重复，所以需要reuse设置从而使用同一套参数，用如下操作
with tf.variable_scope('D_rnn') as scope:
	lstmCell=tf.nn.rnn_cell.BasicLSTMCell(num_units_in_LSTMCell,forget_bias=1.0)
	init_state = lstmCell.zero_state(self.batch_size_t, dtype=tf.float32)
	if reuse:
	     scope.reuse_variables()
	     raw_output, final_state = tf.nn.dynamic_rnn(lstmCell, D_input, initial_state=init_state)
	else:
	      raw_output, final_state = tf.nn.dynamic_rnn(lstmCell, D_input, initial_state=init_state)

3.训练的难点
这次例子在运行进行训练的时候主要存在以下难点
其中D网络的训练速度与提升效果远快于G网络，使得D网络的精度一直提升，而G网络由于对手的精度不断提升的缘故，使得自己的损失越来越大，对抗一边倒。但是G网络尽管损失越来越大，但是经过一定的循环学习之后，输出的数列也开始呈现递增或者递减的规律。但是继续一直学习下去之后输出的这种规律又会弱化。

4.辅助训练技巧
由于生成网络的训练难度较大，所以需要一些符合训练场景的辅助训练技巧。

5 .一些输出变量，例如G,D网络的输出应该使用tf.identity（），保持每一次的输出是当前的最新的计算结果。
'''

=================== code ==============================
import tensorflow as tf
import numpy as np

class GAN_for_timeseq:
	def __init__(self, len_of_seq, lr_d, lr_g):
		# main paramters 
		self.len_of_seq = len_of_seq
		self.lr_d = lr_d
		self.lr_g = lr_g

		# batch_size using placeholder
		self.batch_size_t = tf.placeholder(tf.int32 , shape=[])

		# data flow for g_network using placeholder
		self.g_input_tensor = tf.placeholder(tf.float32 , shape = [None , len_of_seq])
		self.g_input_label = tf.placeholder(tf.float32 , shape = [None , len_of_seq])
		# the output of g_network,and the output is a time_seq
		g_output = self.Generator(self.g_input_tensor)
		self.g_output = tf.reshape(g_output , [-1 , self.len_of_seq , 1])

		# data flow for d_network(here d_network is RNN)
		self.ground_truth_tensor = tf.placeholder(tf.float32 , shape = [None , len_of_seq ,1])
		# the output of d_network when input is ground_truth_tensor
		self.d_output_ground_truth = self.Discriminator(self.ground_truth_tensor,reuse=False)
		self.d_output_fake = self.Discriminator(self.g_output , reuse=True)

		# loss function
		with tf.name_scope('loss'):
			# for g_net,if the time_seq genertaed by it is judged as "true" is more ,the loss would be small
			g_loss_from_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
				logits = self.d_output_fake,
				targets= tf.ones(shape=[tf.reduce_mean(self.batch_size_t) ,1])
				),name = 'g_loss_from_d')  

			self.g_loss = g_loss_from_d

			# for d_net, if the time_seq genertaed by it is judged as "False" is more ,loss is small
			# for d_net, if the ground_truth_tensor is judged as "true" is more, loss is small
			d_loss_from_ground_truth = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
				logits = self.d_output_ground_truth,
				targets = tf.ones(shape = [self.batch_size_t,1])
				#labels = tf.ones(shape=[tf.reduce_mean(self.batch_size_t) ,1])
				),name = 'd_loss_from_ground_truth')
			d_loss_from_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
				logits = self.d_output_fake,
				targets = tf.zeros(shape = [self.batch_size_t,1])
				#labels = tf.zeros(shape=[tf.reduce_mean(self.batch_size_t) , 1])
				))
			self.d_loss = d_loss_from_fake + d_loss_from_ground_truth

		# accuracy about d_net
		with tf.name_scope('accuracy'):
			# accuracy about d_net
			correct_pred_ground_truth = tf.greater(self.d_output_ground_truth , tf.zeros([tf.reduce_mean(self.batch_size_t),1]))
			d_accuracy_ground_truth = tf.reduce_mean(tf.cast(correct_pred_ground_truth , tf.float32))

			correct_pred_fake = tf.less(self.d_output_fake , tf.zeros([tf.reduce_mean(self.batch_size_t),1]))
			d_accuracy_fake = tf.reduce_mean(tf.cast(correct_pred_fake , tf.float32))


		# otimizer operation 
		# get the trainable variables about g_net
		g_net_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'G_net') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'G_rnn') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'G_dnn')
		print(g_net_var_list)
		# train op g_net
		self.train_g_net_op = tf.train.AdamOptimizer(self.lr_g).minimize(self.g_loss,var_list = g_net_var_list)

		# get the trainable variables about d_net
		d_net_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'D_net') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'D_rnn')
		print(d_net_var_list)
		# train op d_net
		self.train_d_net_op = tf.train.AdamOptimizer(self.lr_d).minimize(self.d_loss,var_list = d_net_var_list)

		# model saver
		all_vars = tf.global_variables()
		self.saver = tf.train.Saver(all_vars)

		#######  Session ##################
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())


	# train function
	def train_op(self , batch_size , num_iter , ground_truth_tensor , g_net_input_tensor , g_label_tensor):
		n_samples = ground_truth_tensor.shape[0]
		len_sample = ground_truth_tensor.shape[1]
		n_batch = int(n_samples/batch_size)
		print('n_samples,len_sample,n_batch = ',n_samples,len_sample,n_batch)

		# train loop
		for iter in range(num_iter):
			for batch_id in range(n_batch):
				# make d_net_input batch for d_net
				ground_truth_tensor_batch = ground_truth_tensor[batch_id*batch_size : (batch_id+1)*batch_size]

				# make g_net_input batch for g_net
				if g_net_input_tensor is None:
					g_net_input = np.random.uniform(-1,1,( batch_size,len_sample ))
				else:
					g_net_input = g_net_input_tensor[batch_id*batch_size : (batch_id+1)*batch_size]

				# train for d_net
				feed_dict = {self.batch_size_t:batch_size , 
								self.ground_truth_tensor : ground_truth_tensor_batch , 
								self.g_input_tensor:g_net_input,
								#self.g_input_label : g_input_label
								}
				self.sess.run(self.train_d_net_op , feed_dict = feed_dict)

				# train for g_net
				feed_dict = {self.batch_size_t:batch_size , 
								self.ground_truth_tensor : ground_truth_tensor_batch , 
								self.g_input_tensor:g_net_input,
								#self.g_input_label : g_input_label
								}
				self.sess.run(self.train_g_net_op , feed_dict = feed_dict)

				# info about training
				g_output, d_output_fake, d_output_ground_truth= self.sess.run(
                        [self.g_output, self.d_output_fake, self.d_output_ground_truth], 
                        feed_dict={self.batch_size_t:batch_size , 
							    	self.ground_truth_tensor : ground_truth_tensor_batch , 
									self.g_input_tensor: g_net_input,
									#self.g_input_label : g_input_label
									})
				# loss about two nets
				d_loss_ = self.sess.run(self.d_loss,feed_dict = {self.batch_size_t:batch_size , 
							    	self.ground_truth_tensor : ground_truth_tensor_batch , 
									self.g_input_tensor: g_net_input,
									#self.g_input_label : g_input_label
									})
				g_loss_ = self.sess.run(self.g_loss,feed_dict = {self.batch_size_t:batch_size , 
							    	self.ground_truth_tensor : ground_truth_tensor_batch , 
									self.g_input_tensor: g_net_input,
									#self.g_input_label : g_input_label
									})


			# print info about train
			if iter%10 == 0:
				print('*********************iteration:',iter,'********************')
				print('g_output:',g_output[-1])
				#print('d_output_fake:',d_output_fake)
				#print('d_output_ground_truth:',d_output_ground_truth)
				print('################# GAN loss ##########################')
				print('G LOSS ----->',g_loss_,'       D LOSS----->',d_loss_)
				print('****************************************************************')
				# save model 
				self.save_model('F:\\gan_model_save\\',iter)
	
	# function to make generative network
	def Generator(self , g_input, method = 'dnn'):
		with tf.name_scope('G_net'):
	
			Generator_input = tf.identity(g_input , name = 'input')
			if method =='dnn':
				with tf.variable_scope('G_dnn'):
					# Generator use a DNN model------------------------------------------
					# paramters of fcn
					Nodes_each_layer = self.len_of_seq
					num_layer = 1

					previous_output = Generator_input
					for layer_id in range(num_layer):
						activited_output , output = self.fullConnectedLayer(previous_output , Nodes_each_layer, layer_id)
						previous_output = activited_output
				g_output = output
				g_output = tf.identity(g_output,'g_output')
				return g_output
			else:
				with tf.variable_scope('G_rnn'):
					# Generator use a RNN model ------------------------------------
					# paramters of rnn
					len_seq = int(Generator_input.shape[1])
					time = len_seq
					Generator_input = tf.reshape(Generator_input,[ -1, time , 1 ]) #FEA_NUM =1
					num_units_in_LSTMCell = 10

					lstmCell =  tf.nn.rnn_cell.BasicLSTMCell(num_units_in_LSTMCell)
					init_state = lstmCell.zero_state(self.batch_size_t, dtype=tf.float32)
					raw_output, final_state = tf.nn.dynamic_rnn(lstmCell, Generator_input, initial_state=init_state)
				# cell中出来的数据的维度：【 batch，time ，cell_num】 通过unpack改变成【 time，batch ，cellnum】然后可以获取最后一个时刻的输出
				rnn_output_list = tf.unstack(tf.transpose(raw_output, [1, 0, 2]), name='outList')
				rnn_output_lastmoment = rnn_output_list[-1]
				
				#将最后一个时刻的输出数据输入全连接层,这个矩阵第i行第j列表示这个batch中第i个样本的第j个cell单元输出的数据
				g_sigmoid, g_logit = self.fullConnectedLayer(rnn_output_lastmoment, seq_len , 1)
				g_logit = tf.identity(g_logit, 'g_net_logit')
				return g_logit

	# function to make full connected layer
	def fullConnectedLayer(self, input_tensor , num_node_layer , layer_id):
		layer_id_str = 'fc' + str(layer_id)
		num_input_fea = input_tensor.get_shape()[1].value
		w = tf.Variable(initial_value = tf.random_normal([ num_input_fea , num_node_layer ]) )
		b = tf.Variable(initial_value = tf.zeros([ 1 , num_node_layer ]))
		z = tf.matmul( input_tensor , w ) + b
		z = tf.identity(z )
		activited = tf.nn.relu(z )
		#activited = tf.nn.tanh(z , name = 'activited_'+layer_id_str)
		return activited , z

	# function to make discriminate network
	def Discriminator(self,D_input ,reuse):
		with tf.name_scope('D_net'):
			num_units_in_LSTMCell = 10
            # RNN definition
			with tf.variable_scope('D_rnn') as scope:
				lstmCell = tf.nn.rnn_cell.BasicLSTMCell(num_units_in_LSTMCell,forget_bias=1.0)
				init_state = lstmCell.zero_state(self.batch_size_t, dtype=tf.float32)
				if reuse:
					scope.reuse_variables()
					raw_output, final_state = tf.nn.dynamic_rnn(lstmCell, D_input, initial_state=init_state)
				else:
					raw_output, final_state = tf.nn.dynamic_rnn(lstmCell, D_input, initial_state=init_state)
			rnn_output_list = tf.unstack(tf.transpose(raw_output, [1, 0, 2]), name='outList')
			rnn_output_lastmoment = rnn_output_list[-1]
           	#将最后一个时刻的输出数据输入全连接层,这个矩阵第i行第j列表示这个batch中第i个样本的第j个cell单元输出的数据
			g_sigmoid, g_logit = self.fullConnectedLayer(rnn_output_lastmoment, 1 , 1)
			g_logit = tf.identity(g_logit, 'g_net_logit')
			return g_logit

	def save_model(self , path , step):
		self.saver.save(self.sess , path+'model.ckpt',global_step = step)
		print()
		print('model is saved to',path+'model.ckpt , step is',step)




########################################  test ###########################################
n_samples = 5000
batch_size = 1000
len_seq = 103
num_iter = 5000

ground_truth_tensor = np.ones((n_samples , len_seq)) * np.linspace(1,10,len_seq) + np.random.randn(n_samples,len_seq)*0.5
print(ground_truth_tensor)

n=[]
for i in range(n_samples):
	n.append([i]*len_seq)
m = np.array(n)
print(m)
ground_truth_tensor = ground_truth_tensor + m
ground_truth_tensor = np.reshape(ground_truth_tensor , [-1,len_seq ,1])
noise_tensor = np.random.uniform(-1,1,size = (n_samples,len_seq))

# why we need g_label_tensor??
g_label_tensor = np.random.rand(batch_size , len_seq)

gan = GAN_for_timeseq(len_seq , lr_d = 0.001 , lr_g = 0.01 )
gan.train_op(batch_size , num_iter , ground_truth_tensor , None , None)






            
