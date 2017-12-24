#对训练数据以及测试数据进行预处理

import pandas as pd
import numpy as np
from numpy import matrix
import matplotlib.pyplot as plt
from sklearn.externals import joblib
################################################## 原始数据获取与处理 ###########################
data=pd.read_csv('E:\\kaggle_titanic\\train.csv')
data=data.iloc[:,[1,2,3,4,5,6,7,9,10,11]]#除了name，passageid，ticket外的数据都已经获取
#age和cabin的缺失值标记
data.loc[ (data.Cabin.notnull()), 'Cabin' ] = "Yes"
data.loc[ (data.Cabin.isnull()), 'Cabin' ] = "No"
data.loc[ (data.Age.isnull()), 'Age' ] = "No"

data=data.dropna()
print(data)

#Survived  Pclass  Name   Sex Age  SibSp  Parch      Fare Cabin Embarked

#处理一下Sex特征
for i in range(len(data)):
	if data.iloc[i,3]=='male':
		data.iloc[i,3]=1
	else:
		data.iloc[i,3]=0

#处理一下embarked特征
for i in range(len(data)):
	if data.iloc[i,9]=='S':
		data.iloc[i,9]=1
	elif data.iloc[i,9]=='C':
		data.iloc[i,9]=2
	elif data.iloc[i,9]=='Q':
		data.iloc[i,9]=3

#处理一下Cabin特征
for i in range(len(data)):
	if data.iloc[i,8]=='Yes':
		data.iloc[i,8]=1
	else:
		data.iloc[i,8]=0
print(data)

#处理一下Name特征
for i in range(len(data)):
	if 'Mr.' in data.iloc[i,2]:
		data.iloc[i,2]=0
	elif 'Miss.' in data.iloc[i,2]:
		data.iloc[i,2]=1
	elif 'Mrs.' in data.iloc[i,2]:
		data.iloc[i,2]=2
	elif 'Master.' in data.iloc[i,2]:
		data.iloc[i,2]=3
	elif 'Dr.' in data.iloc[i,2]:
		data.iloc[i,2]=4
	else:
		data.iloc[i,2]=5


	
print(data)
##################################################################################################

################################################ 构造预测年龄的模型 ################################################
#预测年龄的缺失值

age_yes=[]
age_no=[]
for i in range(len(data)):
	if data.iloc[i,4]=='No':
		age_no.append(i)
	else:
		age_yes.append(i)

x=data.iloc[age_yes,[1,2,3,5,6,7,8,9]]
y=data.iloc[age_yes,[4]]
#样本的随机抽样作为测试样本
import random
index=list(range(len(x)))
test_index = random.sample(index, 200)##test_index is the index of test data随机选出2000个样本作为测试样本
train_index=[]
for i in range(len(x)):
	if i not in test_index:
		train_index.append(i)   ##train_index is the index of train data

x_train=x.iloc[train_index,:].as_matrix()
y_train=y.iloc[train_index,:].as_matrix()
x_test=x.iloc[test_index,:].as_matrix()
y_test=y.iloc[test_index,:].as_matrix()

from sklearn.ensemble import GradientBoostingRegressor
gbrt_age=GradientBoostingRegressor( 
  loss='ls'
, learning_rate=0.3
, n_estimators=16 #number of regression trees
, subsample=1
, min_samples_split=1
, min_samples_leaf=1
, max_depth=4 #depth of each individual tree
, init=None
, random_state=None
, max_features=None
, alpha=0.9
, verbose=0
, max_leaf_nodes=None
, warm_start=False
)
gbrt_age.fit(x_train,y_train)
joblib.dump(gbrt_age,'E:\\kaggle_titanic\\gbrt_age.model')

pred=gbrt_age.predict(x_test)
k=0
for i in range(len(x_test)):
	print(pred[i],y_test[i])
	k=k+abs(pred[i]-y_test[i])
print(k/len(x_test))

#进行年龄缺失值补充
for i in age_no:
	data.iloc[i,4]=np.round(gbrt_age.predict(np.array([data.iloc[i,1],data.iloc[i,2],data.iloc[i,3],data.iloc[i,5],data.iloc[i,6],data.iloc[i,7],data.iloc[i,8],data.iloc[i,9]]))[0],0)
print(data)

#fARE归一化
data.Age=(data.Age-data.Age.min())/(data.Age.max()-data.Age.min())
data.Fare=(data.Fare-data.Fare.min())/(data.Fare.max()-data.Fare.min())
'''
data1=pd.DataFrame(np.zeros((len(data),11)))
for i in range(len(data)):
	for j in range(10):
		data1.iloc[i,j]=data.iloc[i,j]
	if data.iloc[i,4]<15:
		data1.iloc[i,10]=1
	else:
		data1.iloc[i,10]=0
'''
#data1.iloc[:,4]=(data1.iloc[:,4]-data1.iloc[:,4].min())/(data1.iloc[:,4].max()-data1.iloc[:,4].min())
data.to_csv('E:\\kaggle_titanic\\clean_train.csv')



##########################################################
# test data 的清洗预处理工作
print('*************************************************')
data=pd.read_csv('E:\\kaggle_titanic\\test.csv')
data=data.iloc[:,[1,2,3,4,5,6,8,9,10]]#除了name，passageid，ticket外的数据都已经获取
print(data)

# Pclass    name  Sex   Age  SibSp  Parch      Fare            Cabin Embarked
#age和cabin的缺失值标记
data.loc[ (data.Cabin.notnull()), 'Cabin' ] = "Yes"
data.loc[ (data.Cabin.isnull()), 'Cabin' ] = "No"
data.loc[ (data.Age.isnull()), 'Age' ] = "No"


#处理一下Sex特征
for i in range(len(data)):
	if data.iloc[i,2]=='male':
		data.iloc[i,2]=1
	else:
		data.iloc[i,2]=0

#处理一下embarked特征
for i in range(len(data)):
	if data.iloc[i,8]=='S':
		data.iloc[i,8]=1
	elif data.iloc[i,8]=='C':
		data.iloc[i,8]=2
	elif data.iloc[i,8]=='Q':
		data.iloc[i,8]=3

#处理一下Cabin特征
for i in range(len(data)):
	if data.iloc[i,7]=='Yes':
		data.iloc[i,7]=1
	else:
		data.iloc[i,7]=0

##处理一下Name特征
for i in range(len(data)):
	if 'Mr.' in data.iloc[i,1]:
		data.iloc[i,1]=0
	elif 'Miss.' in data.iloc[i,1]:
		data.iloc[i,1]=1
	elif 'Mrs.' in data.iloc[i,1]:
		data.iloc[i,1]=2
	elif 'Master.' in data.iloc[i,1]:
		data.iloc[i,1]=3
	elif 'Dr.' in data.iloc[i,1]:
		data.iloc[i,1]=4
	else:
		data.iloc[i,1]=5


print(data)


age_yes=[]
age_no=[]
for i in range(len(data)):
	if data.iloc[i,3]=='No':
		age_no.append(i)
	else:
		age_yes.append(i)
#进行年龄缺失值补充
for i in age_no:
	data.iloc[i,3]=np.round(gbrt_age.predict(np.array([data.iloc[i,0],data.iloc[i,1],data.iloc[i,2],data.iloc[i,4],data.iloc[i,5],data.iloc[i,6],data.iloc[i,7],data.iloc[i,8]]))[0],0)
print(data)

#fARE归一化
data.Age=(data.Age-data.Age.min())/(data.Age.max()-data.Age.min())
data.Fare=(data.Fare-data.Fare.min())/(data.Fare.max()-data.Fare.min())

'''
data1=pd.DataFrame(np.zeros((len(data),10)))
for i in range(len(data)):
	for j in range(9):
		data1.iloc[i,j]=data.iloc[i,j]
	if data.iloc[i,3]<15:
		data1.iloc[i,9]=1
	else:
		data1.iloc[i,9]=0
'''
#data1.iloc[:,3]=(data1.iloc[:,3]-data1.iloc[:,3].min())/(data1.iloc[:,3].max()-data1.iloc[:,3].min())


data.to_csv('E:\\kaggle_titanic\\clean_test.csv')
