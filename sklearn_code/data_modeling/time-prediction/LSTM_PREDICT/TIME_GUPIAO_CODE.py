###########
########
#数据处理方法参考文章《教你打造股市晴雨表——通过LSTM神经网络预测股市》

import pandas as pd
import numpy as np
import random
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Dropout
from  keras.layers.core import RepeatVector, TimeDistributedDense, Activation
import matplotlib.pyplot as plt

num_consider=9
num_predict=2
num_test=52
train_num=50

all_data=pd.read_excel('E:\Master\PPDAMcode\stock\dengji.xls')




###############################################################################################################################
###################################################训练样本准备#####################################################################
from_num=0                      #把第from_num个之后的数据用以训练，模型已经通过多次训练
AQI=(all_data.iloc[:,0]) #从execl的第一列中提取存放进去的历史数据，用来训练网络

l=[]#存放输入量的列表
s=[]#存放输出量的列表
for i in range(len(AQI)-num_predict-num_consider+1): #i循环一次，就是获取一次包含x与y的训练样本 
  ll=[]
  ss=[]
  #列表ll与ss在每个循环中暂时存放同一个训练样本中的x与y

  for j in range(num_predict+num_consider):   #对应每一个i的循环值，通过j获取x和y的所有数据，前num_consider是x的数据
    if j<num_consider:
      ll.append(AQI[i+from_num+j])
    else:
      ss.append(AQI[i+from_num+j])  ######超过num_consider个数据后的数据是y的数据

  s.append(ss)########将形成的变量样本放入列表
  l.append(ll)########将形成的因变量样本放入列表

d=np.asarray(pd.DataFrame(l))
dy=np.asarray(pd.DataFrame(s)) #通过窗口的滑动，从第二个数据到最后一个数据，形成所有的训练样本

d= np.reshape(d, (d.shape[0],d.shape[1],1))#########将所有变量样本x转变为lstm所需要的输入三维格式

for i in range(len(dy)):
  for j in range(len(dy[0])):
    dy[i][j]=(dy[i][j]/d[i][0]-1.0000)    ##########对数据进行标准化转换


for i in range(len(d)):
  for j in range(len(d[0])):
    d[i][num_consider-1-j]=(d[i][num_consider-1-j]/d[i][0]-1.0000) ##########对数据进行标准化转换

####################################################################################################################







####################################################################################################################
##################################################网络搭建############################################################
model_LSTM = Sequential()
 
model_LSTM.add(LSTM(output_dim=100,input_dim=1,activation='tanh',return_sequences=True))

model_LSTM.add(LSTM(output_dim=50,activation='tanh',return_sequences=False))

model_LSTM.add(Dense(output_dim=10,activation='linear'))

model_LSTM.add(Dense(output_dim=num_predict,activation='linear'))

model_LSTM.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

#model_LSTM.load_weights('E:\Master\PPDAMcode\stock\qingdaobeer_LSTM.h5')
#model_LSTM.fit(x=d, y=dy, nb_epoch=train_num, batch_size=20)
model_LSTM.save_weights('E:\Master\PPDAMcode\stock\dengji_LSTM.h5')

#################################################################################################################




################################################################################################################
###############################################测试模型载入########################################################
model_LSTM.load_weights('E:\Master\PPDAMcode\stock\dengji_LSTM.h5')





####################################测试数据准备#########################################################################
AQI_TEST=(all_data.iloc[:num_test,1])  #从execl表中提取第二列数据作为测试数据

l=[]
s=[]
for i in range(len(AQI_TEST)-num_consider):  ##减去的数字是样本中包含数据的个数
  ll=[]
  ss=[]
  for j in range(num_consider):#####该数字是一个样本中包括变量与因变量数据的总个数
    if (i+j)<len(AQI_TEST):
      ll.append(AQI_TEST[i+j])  ##滑动窗口，形成测试数据
  l.append(ll)

d_test0=np.asarray(pd.DataFrame(l))

d0=[]
for i in range(len(d_test0)):
  d0.append(d_test0[i][0])

d_test= np.reshape(d_test0, (d_test0.shape[0],d_test0.shape[1],1))

for i in range(len(d_test)):
  for j in range(len(d_test[0])):
    d_test[i][num_consider-1-j]=(d_test[i][num_consider-1-j]/d_test[i][0]-1.0000)

i=0
ld=[]
lp=[]

p_time=0# 预测次数

for l in d_test:
  if i%num_predict==0:   ######因为是一次性预测7个未来数据，所有避免窗口的重复，所有测试数每7个使用一次
    p_time=p_time+1
    for n in model_LSTM.predict(np.reshape(l, (1,l.shape[0],l.shape[1])))[0]:
      lp.append((n+1)*d0[i])
  i=i+1

for i in range(len(AQI_TEST)-num_consider):#构造测试真实数据对照
  ld.append(AQI_TEST[i+num_consider])
#################################################################################################################





####################################################测试结果曲线##############################################################
print('test......')
color=['b','g','c','m']
time=list(range(len(lp)))#与预测数据时间对应的序号

'''
for j in range(len(lp)):
  lpp=[]
  time_=[]
  if j%(num_predict)==0:
    if len(lp)>(j+num_predict-1):
      lpp.append(lp[j])
      lpp.append(lp[j+num_predict-1])
      time_.append(time[j])
      time_.append(time[j+num_predict-1])
      line_prediction=plt.plot(time_,pd.Series(lpp))
      if lp[j]>lp[j+num_predict-1]:
        plt.setp(line_prediction,color='g',linewidth=2.0)
      else:
        plt.setp(line_prediction,color='r',linewidth=2.0)

for i in range(p_time):#画出预测数据曲线，每一次预测用不同的颜色区别
  line_prediction=plt.plot(time[i*num_predict:(i+1)*num_predict],pd.Series(lp)[i*num_predict:(i+1)*num_predict])
  plt.setp(line_prediction,color=color[i%4],linewidth=3.0)
'''
pd.Series(ld).plot() #直接画出预测数据曲线

pd.Series(lp).plot() #直接画出预测数据曲线

plt.show()

k=0
for i in range(len(lp)-1):
  if ld[i]<50:
    ldd=0
  if 50<ld[i]<100:
    ldd=1
  if 100<ld[i]<150:
    ldd=2
  if 150<ld[i]<200:
    ldd=3
  if ld[i]>200:
    ldd=4

  if lp[i]<50:
    lpp=0
  if 50<lp[i]<100:
    lpp=1
  if 100<lp[i]<150:
    lpp=2
  if 150<lp[i]<200:
    lpp=3
  if lp[i]>200:
    lpp=4
  print(ldd,lpp)
  if ldd==lpp:
    k=k+1
print(k/(len(lp)-1))
