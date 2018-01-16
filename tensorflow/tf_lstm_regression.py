import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#定义一下各个全局变量，在以下整个代码中都使用这个常量，一般用大写

BATCH_START = 0 
TIME_STEPS =10
BATCH_SIZE = 10
BATCH_SIZE_TEST = 1
INPUT_SIZE = 1
OUTPUT_SIZE =1
CELL_SIZE = 10
LR = 0.005
train_step = 500


# 定义data and label
# 实际上的输入只是一个数据元素 ， 但是由于lstm是需要一个时序的输入，所以输入其实是一个time_steps长度的时序数据（一个样本）
# 注意输入输出数据的格式 -> 【 BATCH_SIZE ,TIME_STEPS ， np.newaxis】 三个维度分别表示批量的样本数量 ， 每个样本的时刻数 ， 特征数量
def get_batch(string):
	if string == 'train':
		global BATCH_START , TIME_STEPS, BATCH_SIZE
		x = np.arange(BATCH_START , BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE ,TIME_STEPS ))
		in_seq = np.sin(x)
		out_seq = np.cos(x)
		BATCH_START = BATCH_START + BATCH_SIZE
	return [in_seq[:,:,np.newaxis] , out_seq[:,:,np.newaxis] , x]



# 定义一个LSTM_RNN的类 ， 这个类中包含了各个lstm需要的参数（实例化的时候传入）
# 类在实例化的时候自动按顺序调用了在内部定义的 add_input_layer() ，add_cell() ，add_output_layer() 三个函数，从而搭建起了神经网络的结构
# add_input_layer() ，add_cell() ，add_output_layer() 三个函数都在类的内部定义了
# 按顺序实现了以下过程：输入训练数据-->in_hidden层-->lstm_cell层-->out_hidden层-->前向计算得到的输出数据
# 另外定义了 _weight_variable() , _bias_variable() 函数 ， 用于add_input_layer() ，add_cell() ，add_output_layer()在其内部调用、
# 另外定义了 compute_cost 函数，在类实例化时候做完前向计算后计算误差，计算误差的时候会内部调用函数msr_error()
# 另外在类实例化的时候，会自动调用各种内部定义的函数，在调用的过程当中，会同时建立一些必须的self.的属性
# 包括 self.l_in_y，self.cell_init_state， self.cell_outputs , self.cell_final_state， self.pred，self.cost等
class LSTM_RNN(object):
   def __init__(self, time_steps , input_num_fea , output_num_fea , cell_size , batch_size):
      self.time_steps = time_steps
      self.input_num_fea = input_num_fea
      self.output_num_fea = output_num_fea
      self.cell_size = cell_size
      self.batch_size = batch_size

      with tf.name_scope('inputs'):
         self.xs = tf.placeholder(tf.float32,[None,time_steps,input_num_fea])
         self.ys = tf.placeholder(tf.float32,[None,time_steps,output_num_fea])

      with tf.variable_scope('in_hidden'):
         self.add_input_layer()

      with tf.variable_scope('Lstm_cell'):
         self.add_cell()

      with tf.variable_scope('out_hidden'):
         self.add_output_layer()

      with tf.name_scope('cost'):
         self.compute_cost()

      with tf.name_scope('train'):
         self.train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(self.cost)

   def add_input_layer(self):
      # change the data shape from [batch_size ,time_steps ,input_num_fea ]
      # to [batch_size * time_step , input_num_fea]
      l_in_x = tf.reshape(self.xs , [-1 , self.input_num_fea] , name = '2_2D')

      Weight_in = self._weight_variable([self.input_num_fea,self.cell_size ])
      bs_in = self._bias_variable([self.cell_size])
      with tf.name_scope('Wx_plus_b'):
         l_in_y = tf.matmul(l_in_x , Weight_in) + bs_in
         # get l_in_y ==> [batch_size * time_step , cell_size]

      # change the l_in_y from [batch_size * time_step , cell_size] to [batch_size , time_step , cell_size]
      self.l_in_y = tf.reshape(l_in_y , [-1 , self.time_steps , self.cell_size] , name = '2-3D')

   def add_cell(self):
   	   # get the data shape ==> [batch_size , time_step , cell_size]
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0 , state_is_tuple= True)
      with tf.name_scope('init_state'):
         self.cell_init_state = lstm_cell.zero_state(self.batch_size , tf.float32)
      self.cell_outputs , self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell,self.l_in_y,initial_state=self.cell_init_state , time_major= False)
   	# the cell_outputs from [batch_size , time_steps , cell_size] 

   def add_output_layer(self):
      #change the cell_outputs from [batch_size , time_steps , cell_size] to [batch_size*time_step , cell_size]
      l_output_x = tf.reshape(self.cell_outputs , [-1 , self.cell_size] , name='2_2D')
      Weight_out = self._weight_variable([self.cell_size,self.output_num_fea ])
      bs_out = self._bias_variable([self.output_num_fea])

      with tf.name_scope('Wx_plus_b'):
         self.pred= tf.matmul(l_output_x , Weight_out) + bs_out
         # pred shape ==> [ batch_size*time_step , output_num_fea]


   def compute_cost(self):
      losses = tf.nn.seq2seq.sequence_loss_by_example(
      [tf.reshape(self.pred , [-1 ] , name = 'reshaped_pred')],
      [tf.reshape(self.ys , [-1] , name = 'reshaped_target')],
      [tf.ones([self.batch_size*self.time_steps] ,dtype=tf.float32 )],
      average_across_timesteps= True,
      softmax_loss_function= self.msr_error,
      name = 'losses'
      )

      with tf.name_scope('average_cost'):
          self.cost = tf.div(tf.reduce_sum(losses,name = 'losses_sum'), tf.cast(self.batch_size , tf.float32),name='average_cost')
          tf.scalar_summary('cost',self.cost)

   def msr_error(self , y_pre , y_target):
       return tf.square(tf.sub(y_pre , y_target))

   def _weight_variable(self,shape , name = 'weights'):
       initializer = tf.random_normal_initializer(mean=0, stddev=1.0,)
       return tf.get_variable(shape= shape, initializer=initializer,name =name)

   def _bias_variable(self , shape , name ='biase'):
       initializer = tf.constant_initializer(0.1)
       return tf.get_variable(name=name , initializer=initializer , shape=shape)


数据shape在前向计算过程中的变化



if __name__ == '__main__':
	model = LSTM_RNN(time_steps= TIME_STEPS , input_num_fea= INPUT_SIZE , output_num_fea= OUTPUT_SIZE , cell_size= CELL_SIZE , batch_size= BATCH_SIZE)
	sess = tf.Session()

	merged = tf.merge_all_summaries()
	writer = tf.train.SummaryWriter('e:\\',sess.graph)

	sess.run(tf.initialize_all_variables())

	for step in range(train_step):
		in_seq , out_seq ,x= get_batch('train')
		# 如果是第一次训练的话，那么LSTM的初始cell状态是默认的 self.cell_init_state
		# 如果是之后的训练（之后的batch数据训练的时候），那么初始状态是上一次训练结束获得的finalstate，即model.cell_final_state
		if step == 0:
			feed_dict ={model.xs : in_seq , model.ys : out_seq}
		else:
			feed_dict ={model.xs : in_seq , model.ys : out_seq , model.cell_init_state : state}

		_ ,cost , state ,pred = sess.run( [ model.train_op, model.cost, model.cell_final_state , model.pred ] , feed_dict)
    ############################   
	print('test ......')
	in_seq , out_seq ,x= get_batch('train')
	in_seq=in_seq
	out_seq=out_seq
	print('test_input:')
	print(in_seq)
	print('\ntest_label:')
	print(out_seq)

	feed_dict ={model.xs : in_seq }
	pred = sess.run( [ model.pred ] , feed_dict)
	plt.plot(pred[0].reshape(( 1,-1))[0],c='b',lw=5)
	plt.plot(out_seq.flatten(),c='r',lw=5)
	plt.show()
