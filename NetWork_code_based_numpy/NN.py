#-*- coding: utf-8 -*-
# 手写简单神经网络实现手写数字分类（规模 28*28 ---> 50 ---> 10，输入层784，隐藏层50，输出层10）
#吴一帆

import numpy as np
from scipy import io as spio
import struct
import matplotlib.pyplot as plt

############################################# prepare the minist data for train and test ########
# 读取训练图片的输入数据----------------------------------------------
def read_train_images(filename):
	binfile = open(filename, 'rb')
	buf = binfile.read()
	index = 0
	magic, train_img_num, numRows, numColums = struct.unpack_from('>IIII', buf, index)
	print(magic, ' ', train_img_num, ' ', numRows, ' ', numColums)
	index += struct.calcsize('>IIII')
	train_img_list = np.zeros(( train_img_num,28*28))
	for i in range(train_img_num):
		im = struct.unpack_from('>784B', buf, index)
		index += struct.calcsize('>784B')
		im = np.array(im)
		im = im.reshape(1, 28 * 28)
		train_img_list[ i , : ] = im
	return train_img_list


# 读取训练图片的label数据----------------------------------------------
def read_train_labels(filename):
	binfile = open(filename, 'rb')
	index = 0
	buf = binfile.read()
	binfile.close()

	magic, train_label_num = struct.unpack_from('>II', buf, index)
	index += struct.calcsize('>II')
	train_label_list = np.zeros(( train_label_num,1))
	for i in range(train_label_num):
		# for x in xrange(2000):
		label_item = int(struct.unpack_from('>B', buf, index)[0])
		train_label_list[ i , : ] = label_item
		index += struct.calcsize('>B')
	return train_label_list



# 读取test图片的输入数据----------------------------------------------
def read_test_images(filename):
	binfile = open(filename, 'rb')
	buf = binfile.read()
	index = 0
	magic, test_img_num, numRows, numColums = struct.unpack_from('>IIII', buf, index)
	print(magic, ' ', test_img_num, ' ', numRows, ' ', numColums)
	index += struct.calcsize('>IIII')
	test_img_list = np.zeros(( test_img_num,28*28 ))
	for i in range(test_img_num):
		im = struct.unpack_from('>784B', buf, index)
		index += struct.calcsize('>784B')
		im = np.array(im)
		im = im.reshape(1, 28 * 28)
		test_img_list[i, :] = im
	return test_img_list


# 读取test图片的label数据----------------------------------------------
def read_test_labels(filename):
	binfile = open(filename, 'rb')
	index = 0
	buf = binfile.read()
	binfile.close()

	magic, test_label_num = struct.unpack_from('>II', buf, index)
	index += struct.calcsize('>II')
	test_label_list = np.zeros(( test_label_num , 1))
	for i in range(test_label_num):
		# for x in xrange(2000):
		label_item = int(struct.unpack_from('>B', buf, index)[0])
		test_label_list[i, :] = label_item
		index += struct.calcsize('>B')
	return test_label_list

# minist 的数据直接放在E盘下，进行网络训练或进行测试数据的测试时，直接从E盘读取数据
print('loading train data....')
X_data = read_train_images("E:\\train-images.idx3-ubyte")
Y_data = read_train_labels("E:\\train-labels.idx1-ubyte")
print('loading test data....')
X_test_data = read_test_images("E:\\t10k-images.idx3-ubyte")
Y_test_data = read_test_labels("E:\\t10k-labels.idx1-ubyte")

##############################################################################################
# Neural_Network函数中传入网络的规模参数，以及训练和测试数据
# Neural_Network 中执行的内容： 
#1.初始化神经网络参数  
#2.执行神经网络的优化(通过NN_gradient函数计算出的参数梯度，进行参数的梯度下降优化) 
#3.测试数据的label预测


def Neural_Network(input_size , hidden_layer_size , out_put_size , X,Y,X_test,Y_test):

	m,n = X.shape

	# 初始化神经网络参数---------------------------------------
	initial_weight_1 = random_initial_weight(input_size , hidden_layer_size)
	initial_weight_2 = random_initial_weight(hidden_layer_size , out_put_size)
	initial_nn_weights = np.vstack(( initial_weight_1.reshape(-1,1),initial_weight_2.reshape(-1,1) ))
	
	'''
	# 执行神经网络的优化 -----------------------------------
	lam =0.05 #正则系数
	# cur_nn_weights = initial_nn_weights                  # 从初始化参数开始优化----------
	cur_nn_weights = np.loadtxt("E:\\dnn_WEIGHTS_05.txt")  # 从之前优化得到的参数开始继续优化（增量学习）----------
	
	cur_nn_weights = cur_nn_weights.reshape(-1,1)
	step=100                   #训练次数
	learning_rate = 0.2        #学习率
	num_labels=out_put_size    #输出label个数

	for i in range(step):
		print("TRAIN STEP--->",i)
		grad = NN_gradient(cur_nn_weights , input_size , hidden_layer_size ,  num_labels , X,Y,lam)
		cur_nn_weights = cur_nn_weights - learning_rate*grad.reshape(-1,1)

	result = cur_nn_weights    #通过梯度下降优化得到的参数
	np.savetxt("E:\\dnn_WEIGHTS_05.txt",result)         #save the trained-well weights
	'''
	# 测试数据预测---------------------------------
	weights_load = np.loadtxt("E:\\dnn_WEIGHTS_04.txt") #reload the saved weights
	length_weight = len(weights_load) #从读取进来的参数中抽取出各层的参数
	weight1 = weights_load[0:hidden_layer_size*(input_size+1)].reshape(hidden_layer_size , input_size+1)
	weight2 = weights_load[hidden_layer_size*(input_size+1):length_weight].reshape(out_put_size , hidden_layer_size+1)

	print('predict testing...')
	predict_label = predict(weight1 , weight2 , X_test,Y_test) #根据训练好的参数，对测试数据进行预测
	print("accuracy of prediction is ",np.mean(np.float64(predict_label == Y_test.reshape(-1,1))*100),'%')  #预测正确率
	if len(predict_label) < 20:
		print('predict label  ','real label')
		print(predict_label,'       ',Y_test.reshape(-1,1))
	return np.mean(np.float64(predict_label == Y_test.reshape(-1,1))*100)

#############################################################################################
# NN_cost_function函数，输入为初始参数，网络规模，以及训练数据，正则系数 ，计算出每一次前向传播的损失
def NN_cost_function(initial_nn_weights , input_size , hidden_layer_size , num_labels , X ,Y ,lam):

	length_weight = initial_nn_weights.shape[0]
	weight1 = initial_nn_weights[0:hidden_layer_size*(input_size+1)].reshape(hidden_layer_size , input_size+1)
	weight2 = initial_nn_weights[hidden_layer_size*(input_size+1):length_weight].reshape(num_labels , hidden_layer_size+1)

	m= X.shape[0] #X的行数
	class_y = np.zeros((m,num_labels)) #class_y表示的是一个m行10列的矩阵，每一行表示某一个训练样本的标记的0-1向量表示形式
	
	for i in range(num_labels):
		class_y[:,i] = np.int32(Y == i).reshape(1,-1)

	#计算L2正则化项(这里不考虑疆偏置bias的参数加入到正则化项中)
	weight1_norm = weight1[:,1:weight1.shape[1]]
	weight2_norm = weight2[:,1:weight2.shape[1]]

	#计算L2正则项
	L2_norm_term = np.dot(  np.transpose(np.vstack((weight1_norm.reshape(-1,1),weight2_norm.reshape(-1,1)))) ,  np.vstack((weight1_norm.reshape(-1,1),weight2_norm.reshape(-1,1)))  )

	#前向传播过程
	a1 = np.hstack((np.ones((m,1)), X)) #在每个训练样本的输入中加入一个偏置项 1
	z2 =np.dot(a1 , np.transpose(weight1))

	a2 = sigmoid(z2)

	a2 = np.hstack(( np.ones((m,1)),a2 ))
	z3 = np.dot(a2 , np.transpose(weight2))

	output = sigmoid(z3)

	# 误差代价函数 (交叉熵加上正则项)
	J_error = -np.dot( np.transpose(class_y.reshape(-1,1)), np.log(output.reshape(-1,1))) - np.dot( np.transpose(1-class_y.reshape(-1,1)), np.log(1 - output.reshape(-1,1)))
	J_error = J_error + lam*L2_norm_term/2
	J_error = J_error/m

	return np.ravel(J_error)


###########################################################################################################
# NN_gradient 函数 输入待更新参数，网络结构规模，以及训练数据，正则系数， 计算出每个参数的更新梯度
def NN_gradient(initial_nn_weights , input_size , hidden_layer_size ,  num_labels , X,Y,lam):
	length_weight = initial_nn_weights.shape[0]

	weight1 = initial_nn_weights[0:hidden_layer_size*(input_size+1)].reshape(hidden_layer_size , input_size+1)
	weight2 = initial_nn_weights[hidden_layer_size*(input_size+1):length_weight].reshape(num_labels , hidden_layer_size+1)

	m= X.shape[0] #X的行数
	class_y = np.zeros((m,num_labels)) #class_y表示的是一个m行10列的矩阵，每一行表示某一个训练样本的标记的0-1向量表示形式
	
	for i in range(num_labels):
		class_y[:,i] = np.int32(Y == i).reshape(1,-1)
	
	#抽取出需要更新的参数(这里不考虑疆偏置bias的参数)
	weight1_need_update = weight1[:,1:weight1.shape[1]]
	weight2_need_update = weight2[:,1:weight2.shape[1]]
	 
	weight1_need_update_gradient = np.zeros((weight1.shape))
	weight2_need_update_gradient = np.zeros((weight2.shape))

	weight1[:,0] = 0
	weight2[:,0] = 0

	#正向传播
	a1 = np.hstack((np.ones((m,1)), X)) #在每个训练样本的输入中加入一个偏置项 1
	z2 =np.dot(a1 , np.transpose(weight1))
	a2 = sigmoid(z2)
	a2 = np.hstack(( np.ones((m,1)),a2 ))
	z3 = np.dot(a2 , np.transpose(weight2))
	output = sigmoid(z3)

	#反向传播
	error_out = np.zeros((m,num_labels))
	error_hidden = np.zeros(( m,hidden_layer_size))

	for i in range(m):

		error_out[i,:] = output[i,:] - class_y[i,:] #计算每个样本的输出error
		weight2_need_update_gradient = weight2_need_update_gradient + np.dot(np.transpose(error_out[i,:].reshape(1,-1)) , a2[i,:].reshape(1,-1))

		error_hidden[i,:] = np.dot(error_out[i,:].reshape(1,-1) , weight2_need_update)*sigmoid_gradient(z2[i,:])
		weight1_need_update_gradient = weight1_need_update_gradient + np.dot(np.transpose(error_hidden[i,:].reshape(1,-1)) , a1[i,:].reshape(1,-1))

		# gradient
		grad = (np.vstack(( weight1_need_update_gradient.reshape(-1,1),weight2_need_update_gradient.reshape(-1,1 ))))/m 
		grad = grad + lam*np.vstack((weight1.reshape(-1,1) , weight2.reshape(-1,1)))/m

	return np.ravel(grad)

#################################################################################################################
# 定义上述过程中需要用到的函数

def sigmoid(z):
	output = np.zeros(( len(z),1 ))
	output = 1.0/(1.0+np.exp(-z))
	return output

def sigmoid_gradient(z):
	g=sigmoid(z)*(1-sigmoid(z))
	return g

def tanh(z):
	return np.tanh(z)

def tanh_gradient(z):
    return 1.0 - np.tanh(z)*np.tanh(z)

def random_initial_weight(L_in,L_out):
	W=np.random.randn(L_out, L_in+1)*0.05
	return W

def predict( weight1 , weight2 , X_test , Y_test):
	print('x_test.shape-->',X_test.shape)
	m=X_test.shape[0]
	num_labels = weight2.shape[0]

	#执行预测过程中的正向传播过程
	X_test=np.hstack((np.ones((m,1)) , X_test))
	a1=X_test
	z2=np.dot(a1,np.transpose(weight1))
	a2=sigmoid(z2)
	a2 = np.hstack((np.ones((m,1)) , a2))
	z2 = np.dot(a2,np.transpose(weight2))
	output = sigmoid(z2)
	
	#返回每一行结果中最大概率所在列标号
	pred = np.array(np.where(output[0,:] == np.max(output , axis=1)[0]))
	for i in range(1,m):
		t=np.array(np.where(output[i,:] == np.max(output , axis=1)[i])) #返回output第i行中最大数值所在的列index
		pred=np.vstack((pred,t))
	#print(pred)
	return pred

###############################################################################################################
def check_gradient():
	#通过构造一个相对来说比较小的神经网络来检查梯度
	input_size=5
	hidden_layer_size=10
	out_put_size=1
	num_labels=1
	m=1000
	initial_weight1 = random_initial_weight(input_size , hidden_layer_size)
	initial_weight2 = random_initial_weight(hidden_layer_size , out_put_size)
	X=input_initial_weight(m,input_size)*100
	Y= 1+np.transpose(np.mod(np.arange(1,m+1) , num_labels))
	Y=Y.reshape(-1,1)
	initial_nn_weights = np.vstack(( initial_weight1.reshape(-1,1),initial_weight2.reshape(-1,1) ))
	
	# get gradient using bp method
	lam=0
	grad = NN_gradient(initial_nn_weights , input_size , hidden_layer_size , num_labels, X,Y,lam)

	# get gradient using math defination
	num_grad = np.zeros((initial_nn_weights.shape[0]))
	delta = np.zeros((initial_nn_weights.shape[0]))
	e=1e-4
	for i in range(initial_nn_weights.shape[0]):
		delta[i]=e
		J_error1 = NN_cost_function(initial_nn_weights - delta.reshape(-1,1) , input_size , hidden_layer_size , num_labels , X,Y ,lam)
		J_error2 = NN_cost_function(initial_nn_weights + delta.reshape(-1,1) , input_size , hidden_layer_size , num_labels, X,Y ,lam)

		num_grad[i] = (J_error2 - J_error1)/(2*e)
		delta[i] = 0

	check_grad = np.hstack(( grad.reshape(-1,1),num_grad.reshape(-1,1) ))
	print('checking gradient.....')
	for i in range(check_grad.shape[0]):
		print(abs(check_grad[i,0] - check_grad[i,1])/(max(check_grad[i,0] , check_grad[i,1])))

def input_initial_weight(m,n):
	return np.random.random((m,n))*0.25

############################################## running ################################################################ 
check_gradient()
'''
# 训练样本
X=X_data[:60000,:]
Y=Y_data[:60000,:]

# 所有测试样本
X2=X_test_data[:10000,:]
Y2=Y_test_data[:10000,:]

# 选取单个测试数据
num_picture = 9000
X_test_single = X_test_data[num_picture:num_picture+1,:] #验证所有
Y_test_single = Y_test_data[num_picture:num_picture+1,:]

# 运行
Neural_Network(28*28 , 50 , 10 , X,Y,X_test_single,Y_test_single )

'''



 






	















