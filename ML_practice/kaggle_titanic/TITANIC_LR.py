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

############################# 构造是否生还预测模型 #############################################
#logistic模型-----------------------------------------------------------
from sklearn.linear_model import LogisticRegression as LR
lr=LR(
penalty='l2', 
dual=False, 
tol=0.001, 
C=100.0, 
fit_intercept=True, 
intercept_scaling=1, 
class_weight=None,
random_state=None, 
solver='liblinear', 
max_iter=100, 
multi_class='ovr',
verbose=0, 
warm_start=False,
n_jobs=1)

lr.fit(x_train,y_train)
pred=lr.predict(x_test)

k=0
for i in range(len(x_test)):
	if pred[i]==y_test[i]:
		k=k+1
print(k/len(x_test))

#logistic--bagging模型-----------------------------------------------------------
from sklearn.ensemble import BaggingClassifier
bagging=BaggingClassifier(
base_estimator=lr,
n_estimators=100, 
max_samples=0.9, 
max_features=0.9, 
bootstrap=True, 
bootstrap_features=False, 
oob_score=False, 
warm_start=False, 
n_jobs=1, 
random_state=1, 
verbose=0)

bagging.fit(x_train,y_train)
pred=bagging.predict(x_test)

k=0
for i in range(len(x_test)):
	if pred[i]==y_test[i]:
		k=k+1
print(k/len(x_test))

#logistic--adaboost模型-----------------------------------------------------------
from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(
base_estimator=lr, 
n_estimators=100, 
learning_rate=0.7, 
algorithm='SAMME.R', 
random_state=None)

ada.fit(x_train,y_train)
pred=ada.predict(x_test)

k=0
for i in range(len(x_test)):
	if pred[i]==y_test[i]:
		k=k+1
print(k/len(x_test))



###########################################################################################
############################################################################################
################################### 未知数据的预测 ##########################################

data=pd.read_csv("e:\\kaggle_titanic\\clean_test.csv")


#给预测输入集合添加新特征
new_x=np.ones((len(data),n_tree+9))
data=data.as_matrix()
for i in range(len(new_x)):
	new_x[i][0]=data[i][0]
	new_x[i][1]=data[i][1]
	new_x[i][2]=data[i][2]
	new_x[i][3]=data[i][3]
	new_x[i][4]=data[i][4]		
	new_x[i][5]=data[i][5]
	new_x[i][6]=data[i][6]
	new_x[i][7]=data[i][7]
	new_x[i][8]=data[i][8]
node=gbdt_feature.apply(np.array(data))
for i in range(len(node)):
	for j in range(n_tree):
		new_x[i][j+9]=(node[i][j]-3)/11#  这是加入了gbdt构造出来的新特征的训练特征 

##############################################################
#利用RF模型进行预测

pred1=lr.predict(new_x)
pred2=bagging.predict(new_x)
pred3=ada.predict(new_x)

sample=pd.read_csv('E:\\kaggle_titanic\\gender_submission.csv')
for i in range(len(sample)):
	sample.iloc[i,1]=pred3[i]
#print(sample)
sample.to_csv('E:\\kaggle_titanic\\lr_ada_submission.csv')
