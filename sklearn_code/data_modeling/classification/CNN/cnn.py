# coding=utf-8
import numpy as np
import random
import os
import numpy as np
from PIL import Image 
#from keras.utils.visualize_util import plot
#########################准备训练图形样本，转换为像素数据矩阵28*28
imgs=os.listdir('e:\mnist')
data=np.empty((42000,1,28,28),dtype='float32')
data_test=np.empty((1,1,28,28),dtype='float32')
label=np.empty((42000,),dtype='float32')
num=len(imgs)
for i in range(num):
	img=Image.open('E:\mnist\%s'%imgs[i])
	arr=np.asarray(img,dtype='float32')
	data[i,:,:,:]=arr
	label[i]=int(imgs[i].split('.')[0])
print(imgs[0])

imgss=os.listdir('e:\\testpicture')
for i in range(1):
	img=Image.open('E:\\testpicture\\%s'%imgss[i])
	print(img)
	arr=np.asarray(img,dtype='float32')
	print(arr)
	data_test[i,:,:,:]=arr
######################################导入CNN模型建立所需库与相关模块
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
#############################将样本标签转换为向量表示，包含一共有42000个向量
label = np_utils.to_categorical(label, 10)
#print(label)
##############################建立一个神经网络模型框架
model = Sequential()
#----------------------------------------------------------------------------------------------------------------------------------------
#第一个卷积层，4个卷积核，每个卷积核大小5*5。1表示输入的图片的通道,灰度图为1通道。
model.add(Convolution2D(4, 5, 5, border_mode='same', input_shape=(1, 28, 28),dim_ordering="th"))
#激活函数用tanh
model.add(Activation('tanh'))

#--------------------------------------------------------------------------------------------------------------------------------------------
#第二个卷积层，8个卷积核，每个卷积核大小3*3。4表示输入的特征图个数，等于上一层的卷积核个数
model.add(Convolution2D(8,3,3,border_mode='same',dim_ordering="th"))
model.add(Activation('tanh')) 
#采用maxpooling，poolsize为(2,2)
model.add(MaxPooling2D(pool_size=(2, 2)))


#--------------------------------------------------------------------------------------------------------------
#第三个卷积层，16个卷积核，每个卷积核大小3*3
model.add(Convolution2D(16, 3, 3,border_mode='same',dim_ordering="th")) 
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#-----------------------------------------------------------------------------------------------------------
#全连接层，先将前一层输出的二维特征图flatten为一维的。
model.add(Flatten())
#Dense就是隐藏层。16就是上一层输出的特征图个数。4是根据每个卷积层计算出来的：(28-5+1)得到24,(24-3+1)/2得到11，(11-3+1)/2得到4
#model.add(Dense(16*4*4,128))
model.add(Dense(output_dim=128, input_dim=16*4*4, init='normal',activation='tanh'))


#-----------------------------------------------------------------------------------------------------------------
#Softmax分类，输出是10类别
#model.add(Dense(128, 10, init='glorot_normal'))
#model.add(Activation('softmax'))
model.add(Dense(output_dim=10, input_dim=128, init='normal', activation='softmax'))

#############
#开始训练模型
##############
#使用SGD + momentum
#model.compile里的参数loss就是损失函数(目标函数)
#sgd = SGD(l2=0.0,lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,class_mode="categorical")

#调用fit方法，就是一个训练过程. 训练的epoch数设为10，batch_size为100．
#数据经过随机打乱shuffle=True。verbose=1，训练过程中输出的信息，0、1、2三种方式都可以，无关紧要。show_accuracy=True，训练时每一个epoch都输出accuracy。
#validation_split=0.2，将20%的数据作为验证集。
'''
model.fit(data, label, batch_size=100,nb_epoch=10,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.005)

model.save_weights('e:\my_cnn_model_weights.h5')
'''

model.load_weights('e:\my_cnn_model_weights.h5')

pre=model.predict(data_test)
for i in range(1):
	print(np.where(pre[i]==np.max(pre[i])))
#plot(model, to_file='e:\cnnmodel1.png', show_shapes=True)