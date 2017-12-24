import pandas as pd
import numpy as np
from numpy import matrix
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from IPython.display import Image
################################################## 原始train数据获取与处理 ###########################

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
n_tree=2
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
############################# 构造是否生还预测模型 #############################################
#decision tree模型-----------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier 
dtc=DecisionTreeClassifier(
criterion='entropy',
splitter='best',
max_depth=4,
min_samples_split=2, 
min_samples_leaf=1,
min_weight_fraction_leaf=0.0, 
max_features=None, 
random_state=None,
max_leaf_nodes=None, 
min_impurity_split=1e-07, 
class_weight=None, 
presort=False)

dtc.fit(x_train,y_train)
pred=dtc.predict(x_test)

k=0
for i in range(len(x_test)):
	if pred[i]==y_test[i]:
		k=k+1
print(k/len(x_test))

#random forest模型-----------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(
n_estimators=100, 
	#criterion=None,
	max_depth=5,
	min_samples_split=2,
	min_samples_leaf=1,
	min_weight_fraction_leaf=0.0, 
	max_features=9, 
	max_leaf_nodes=None, 
	min_impurity_split=1e-07,
	bootstrap=True, 
	oob_score=False, 
	n_jobs=1, 
	random_state=None, 
	verbose=0, 
	warm_start=False)

rf.fit(x_train,y_train)
pred=rf.predict(x_test)

k=0
for i in range(len(x_test)):
	if pred[i]==y_test[i]:
		k=k+1
print(k/len(x_test))

#GBDT模型-----------------------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier
gbdt=GradientBoostingClassifier(
  loss='deviance'
, learning_rate=0.3
, n_estimators=5 #number of regression trees
, subsample=1.0
, min_samples_split=2
, min_samples_leaf=1
, max_depth=3 #depth of each individual tree
, init=None
, random_state=None
, max_features=1.0
, verbose=0
, max_leaf_nodes=None
, warm_start=True 
)	

gbdt.fit(x_train,y_train)
pred=gbdt.predict(x_test)

k=0
for i in range(len(x_test)):
	if pred[i]==y_test[i]:
		k=k+1
print(k/len(x_test))