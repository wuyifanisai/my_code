import pandas as pd
import numpy as np
from numpy import matrix
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from IPython.display import Image
################################################## 原始train数据获取与处理 ###########################

print(list(range(1,10))+list(range(10,19)))
C=1/0
data=pd.read_csv('E:\\kaggle_titanic\\clean_train.csv')
print(data)


############################################## 构造预测是否生还的DATA ###################################################
x=data.iloc[:,[1,2,3,4,5,6,7,8,9]]
y=data.iloc[:,0]
#---------------------------------------
'''
#样本的随机抽样作为测试样本
import random
index=list(range(len(x)))
test_index = random.sample(index, 300)##test_index is the index of test data随机选出2000个样本作为测试样本
'''
train_index=list(range(300,len(x)))
test_index=[]
for i in range(len(x)):
	if i not in train_index:
		test_index.append(i)   ##train_index is the index of train data


x_train=pd.DataFrame(x).iloc[train_index].as_matrix()
y_train=pd.DataFrame(y).iloc[train_index].as_matrix()

x_test=x.iloc[test_index].as_matrix()
y_test=y.iloc[test_index].as_matrix()
'''
######################### 利用gbrt构造新特征 ##############################################################
n_tree=1
from sklearn.ensemble import GradientBoostingClassifier
gbdt_feature=GradientBoostingClassifier(
  loss='deviance'
, learning_rate=0.4
, n_estimators=n_tree #number of regression trees
, subsample=1
, min_samples_split=2
, min_samples_leaf=1
, max_depth=3 #depth of each individual tree
, init=None
, random_state=None
, max_features=None
, verbose=0
, max_leaf_nodes=None
, warm_start=False 
)	
gbdt_feature.fit(x_train,y_train)

#给训练输入集合添加新特征------------------------------------------
new_x_train=np.ones((len(x_train),n_tree+9))  #含有新增特征以及重复记录的训练输入

for i in range(len(x_train)):
	for j in range(9):
		new_x_train[i][j]=x_train[i][j]

node=gbdt_feature.apply(np.array(x_train))
for i in range(len(x_train)):
	for j in range(n_tree):
		new_x_train[i][j+9]=(node[i][j]-3)/11#  这是加入了gbdt构造出来的新特征的训练特征 
#给测试数据输入集合添加新特征----------------------------------------------------------------------------------------------------
new_x_test=np.ones((len(x_test),n_tree+9))  #含有新增特征以及重复记录的训练输入

for i in range(len(x_test)):
	for j in range(9):
		new_x_test[i][j]=x_test[i][j]

node=gbdt_feature.apply(np.array(x_test))
for j in range(n_tree):
	new_x_test[i][j+9]=(node[i][j]-3)/11#  这是加入了gbdt构造出来的新特征的训练特征 

x_train=new_x_train
x_test=new_x_test
'''



from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.utils import np_utils
import numpy as np


###################################################################
model = Sequential()
model.add(Dense(output_dim=50, input_dim=9, activation='relu'))
model.add(Dense(output_dim=100, input_dim=50, activation='relu'))
model.add(Dense(output_dim=30, input_dim=100, activation='relu'))
model.add(Dense(output_dim=1, input_dim=30, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

model.fit(x_train, y_train, nb_epoch = 10, batch_size = 20) 

pred=model.predict_classes(x_test).reshape(len(x_test)) 

k=0
for i in range(len(x_test)):
	print(pred[i],y_test[i])
	if pred[i]==y_test[i]:
		k=k+1
print(k/len(x_test))
