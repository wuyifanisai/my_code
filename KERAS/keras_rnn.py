from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN , Activation , Dense
from keras.optimizers import Adam

import numpy as np
（模型说明：
用RNN模型来做mnist数据集的分类问题
因为每个图片的数据是28*28
所以疆模型RNN神经网络的输入设为28
也就是说每一个时刻输入的数据是28个，也就是把图片的每一行28个像素作为每一个时刻RNN 的输入，也就是INPUT_SIZE
另外RNN需要设定一个循环的长度，也就是TIME_SIZE，那么把循环次数设定28，刚好是一张图片的像素行数
也就是说RNN循环28次后刚好学习一张图片的所有像素数据


另外批量处理每次处理50张
）

# some parameters of RNN
TIME_STEPS = 28  # SAME MEANING AS HEIGHT OF IMG
INPUT_SIZE = 28  # SAME MEANING AS WIDTH OF IMG
BATCH_SIZE = 50 
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001
BATCH_INDEX = 0

# Download the mnist data
(x_train , y_train),(x_test , y_test) = mnist.load_data()

# data - preprocessing 
x_train = x_train.reshape(( -1,28 ,28 ))/225.0
x_test = x_test.reshape((-1,28,28))/225.0

y_train = np_utils.to_categorical(y_train , nb_classes = 10)
y_test = np_utils.to_categorical(y_test , nb_classes = 10)

# building RNN model
model =Sequential()

# RNN cell
 # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
model.add(SimpleRNN(batch_input_shape= (None ,TIME_STEPS ,INPUT_SIZE) , 
					output_dim = CELL_SIZE ,
return_sequences=False,      # True: output at all steps. False: output as last step.
   			                   stateful=False,              # False： the final state of batch1 is NOT feed into the initial state of batch2 
					))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

#optimizer
model.compile(optimizer = Adam('relu') , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

#training ----
train_cost_list=[]
test_cost_list=[]
test_accuarcy=[]
for step in range(4000):
	x_batch = x_train[ BATCH_INDEX:BATCH_SIZE+BATCH_INDEX , : , :]
	y_batch = y_train[ BATCH_INDEX:BATCH_SIZE+BATCH_INDEX , : ]

	cost = model.train_on_batch(x_batch , y_batch)
	BATCH_INDEX = BATCH_INDEX + BATCH_SIZE
	BATCH_INDEX = 0 if BATCH_INDEX >= x_train.shape[0] else BATCH_INDEX

	if step%10 == 0:
		train_cost_list.append(cost)
		cost1 , accuracy = model.evaluate(x_test , y_test, batch_size=y_test.shape[0], verbose=False)
		print('test cost and accuracy==>',cost1 , accuracy)
		test_cost_list.append(cost1)
		test_accuarcy.append(accuracy)
print(len(train_cost_list))
#plt.plot(train_cost_list,c='r')
plt.plot(test_cost_list,c='b')
plt.plot(test_accuarcy,c='g')
plt.show()












