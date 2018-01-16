关于keras中，利用LSTM做回归预测，一般是做时序预测，就是类似于自回归预测，根据历史窗口数据预测接下来的时刻的数据。这个简单例子的数据是输入是正弦，输出是同一时刻的余弦。
主要注意几个参数：batch_size ， time_size，input_size，cell_size
batch_size表示进行批量训练的时候， 每一批中包含的训练样本数量
time_size表示RNN（LSTM）中每一个神经元的循环次数，就是说每一个神经元可以记忆的时刻数量，也就说训练的时候考虑的序列长为time_size
input_size表示输入到模型中的特征数量，feature的个数，一般类似自回归的预测问题，输入的样本特征数量是1，而通过time_size的设置，其实模型通过神经元的循环，学习考虑了长度为time_size的输入序列
cell_size独立于上述的三个参数，表示神经元的个数，各个神经元之间没有联系，相互独立，而每个神经元可以与不同时刻的自身进行参数信息联系。
数据输入格式：
虽然input_size为1，但不表示输入的数据每一个样本只有一个数值，每一个样本有time_size个数据，构成一个样本（输入数据中的一行），而这个输入数据有多少行根据batch_size来确定，也就说每一批训练有多少个样本。
所以最终给模型训练的x与y都是序列数据，长度一致。
而模型进行预测输入也是定长的序列
在构造输入输出数据集的时候，需要将数据构造成三维模式，三个坐标分别为
batch_size，time_size,feature_num
如下：
xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

-----------------------------------------------------------------------------------------------------------------------------
# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly

# 8 - RNN LSTM Regressor example

# to try tensorflow, un-comment following two lines
import os
os.environ['KERAS_BACKEND']='theano'
import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam

BATCH_START = 0
TIME_STEPS = 5
BATCH_SIZE = 1
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

model = Sequential()
# build a LSTM RNN
model.add(LSTM(
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    return_sequences=True,      # True: output at all steps. False: output as last step.
    stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
))
# add output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='mse',)

print('Training ------------')
for step in range(501):
    # data shape = (batch_num, steps, inputs/outputs)
	X_batch, Y_batch, xs = get_batch()
	cost = model.train_on_batch(X_batch, Y_batch)
	pred = model.predict(X_batch)
	print(X_batch, Y_batch,pred)
	exit()
	plt.plot(xs[0, :], Y_batch[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
	plt.ylim((-1.2, 1.2))
	plt.draw()
	plt.pause(0.1)
	if step % 10 == 0:
	    print('train cost: ', cost)


参考文章：https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/