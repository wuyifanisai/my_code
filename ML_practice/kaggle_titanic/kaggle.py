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
test_index = list(range(100,200))+list(range(300,400))+list(range(400,500))+list(range(700,800))
train_index=[]

for i in range(len(x)):
	if i not in test_index:
		train_index.append(i)   ##train_index is the index of train data


x_train=pd.DataFrame(x).iloc[train_index].as_matrix()
y_train=pd.DataFrame(y).iloc[train_index].as_matrix()

x_test=x.iloc[test_index].as_matrix()
y_test=y.iloc[test_index].as_matrix()



'''

######################### 利用gbrt构造新特征 ##############################################################
n_tree=3
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
new_x_train=np.ones((len(x_train),n_tree+8))  #含有新增特征以及重复记录的训练输入

for i in range(len(x_train)):
	for j in range(8):
		new_x_train[i][j]=x_train[i][j]

node=gbdt_feature.apply(np.array(x_train))
for i in range(len(x_train)):
	for j in range(n_tree):
		new_x_train[i][j+8]=node[i][j]#  这是加入了gbdt构造出来的新特征的训练特征 
#给测试数据输入集合添加新特征----------------------------------------------------------------------------------------------------
new_x_test=np.ones((len(x_test),n_tree+8))  #含有新增特征以及重复记录的训练输入

for i in range(len(x_test)):
	for j in range(8):
		new_x_test[i][j]=x_test[i][j]

node=gbdt_feature.apply(np.array(x_test))
for j in range(n_tree):
	new_x_test[i][j+8]=node[i][j]#  这是加入了gbdt构造出来的新特征的训练特征 

x_train=new_x_train
x_test=new_x_test
'''
############################# 构造是否生还预测模型 #############################################
#-----------------------------------------------------------
#利用GBRT模型建模
kk1=0
print('******************************')
for i in range(100):
	from sklearn.ensemble import GradientBoostingClassifier
	gbdt=GradientBoostingClassifier(
	loss='deviance'
	, learning_rate=0.5
	, n_estimators=25 #number of regression trees
	, subsample=1
	, min_samples_split=1
	, min_samples_leaf=1
	, max_depth=3 #depth of each individual tree
	, init=None
	, random_state=None
	, max_features=None
	, verbose=0
	, max_leaf_nodes=None
	, warm_start=False 
	)	
	gbdt.fit(x_train,y_train)


	pred=gbdt.predict(x_test)
	k=0	
	for i in range(len(x_test)):
		if pred[i]==y_test[i]:
			k=k+1
	kk1=kk1+k/len(x_test)


#-----------------------------------------------------------
#利用RF模型建模
kk2=0
print('***************************')
for i in range(100):
#-----------------------------------------------------------
#利用RF模型建模
	from sklearn.ensemble import RandomForestClassifier
	rfc=RandomForestClassifier(n_estimators=25, max_depth=3)
	rfc.fit(x_train,y_train)	

	pred=rfc.predict(x_test)
	k=0
	for i in range(len(x_test)):
		if pred[i]==y_test[i]:
			k=k+1
	kk2=kk2+k/len(x_test)

print('gbrt_avg:',kk1/100)
print('RF_avg:',kk2/100)

#-------------------------------------
#bagging
from sklearn.ensemble import BaggingClassifier

gbdt0=GradientBoostingClassifier(
	loss='deviance'
	, learning_rate=0.5
	, n_estimators=9 #number of regression trees
	, subsample=1
	, min_samples_split=1
	, min_samples_leaf=1
	, max_depth=3 #depth of each individual tree
	, init=None
	, random_state=None
	, max_features=None
	, verbose=0
	, max_leaf_nodes=None
	, warm_start=False 
	)	

bagging = BaggingClassifier(gbdt0, n_estimators=20,max_samples=0.8, max_features=1.0)
bagging.fit(x_train,y_train)
pred=bagging.predict(x_test)
k=0
for i in range(len(x_test)):
	if pred[i]==y_test[i]:
		k=k+1
print(k/len(y_test))
#------------------------------------
#利用adboost
from sklearn.tree import DecisionTreeClassifier as DTC
dtc=DTC(criterion='entropy')
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(base_estimator=dtc,n_estimators=20)
ada.fit(x_train,y_train)
pred=ada.predict(x_test)
k=0
for i in range(len(x_test)):
	if pred[i]==y_test[i]:
		k=k+1
print(k/len(y_test))
###########################################################################################
############################################################################################
################################### 未知数据的预测 ##########################################

data=pd.read_csv("e:\\kaggle_titanic\\clean_test.csv")
'''
#########################################################################
#给预测输入集合添加新特征
new_x=np.ones((len(data),n_tree+8))
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
node=gbdt_feature.apply(np.array(data))
for i in range(len(node)):
	for j in range(n_tree):
		new_x[i][j+8]=node[i][j]#  这是加入了gbdt构造出来的新特征的训练特征 
'''
##############################################################
#利用RF模型进行预测

x=data
print(data)
pred1=rfc.predict(x)
pred2=gbdt.predict(x)
pred3=bagging.predict(x)
pred4=ada.predict(x)

sample=pd.read_csv('E:\\kaggle_titanic\\gender_submission.csv')
for i in range(len(sample)):
	sample.iloc[i,1]=pred3[i]
#print(sample)
sample.to_csv('E:\\kaggle_titanic\\bagging_submission.csv')


'''
训练模型所用训练数据的多少可能会影响最终的成绩，用了全部数据可能不是最优的方案


'''