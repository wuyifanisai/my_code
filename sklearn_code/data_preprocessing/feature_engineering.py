import numpy as np
from numpy import array
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt


all_data=pd.read_excel('E:\Master\PPDAMcode\AIR_project\hangzhou_air_alldata01.xls')


names=[1,2,3]
import random
index=list(range(1000))
test_index = random.sample(index, 200)##test_index is the index of test data随机选出200个样本作为测试样本
train_index=[]
for i in range(1000):
	if i not in test_index:
		train_index.append(i)   ##train_index is the index of train data

x_train=all_data.iloc[train_index,0:6].as_matrix()
y_train=all_data.iloc[train_index,6].as_matrix()

x_test=all_data.iloc[test_index,0:6].as_matrix()
y_test=all_data.iloc[test_index,6].as_matrix()

y_AQI_train=all_data.iloc[train_index,7].as_matrix()
y_AQI_test=all_data.iloc[test_index,7].as_matrix()
#-----------------------------------------------------------
from sklearn.feature_selection import VarianceThreshold
#print(VarianceThreshold(threshold=0.02).fit_transform(x_train))
#根据方差选择特征，选择方差大于阈值的特征
#-------------------------------------------------------------
def feature_selection_pearsonr(x_train,y_AQI_train,threshold):
	from scipy.stats import pearsonr
	num=[]
	for i in range(len(x_train.T)):
		print(pearsonr(x_train.T[i],y_AQI_train))
		if pearsonr(x_train.T[i],y_AQI_train)[0]>threshold:
			num.append(i)
	return x_train[:,num]
#print(feature_selection_pearsonr(x_train,y_AQI_train,0.8))
#通过计算每一个特征和标签的相关性系数，选择相关性系数大于阈值的特征
#-----------------------------------------------------------------
#经典卡方检验
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2#选择K个最好的特征，返回选择特征后的数据
#print(SelectKBest(chi2, k=2).fit_transform(x_train,y_AQI_train))
#----------------------------------------------------------------------
#递归消除特征法使用一个基模型来进行多轮训练，每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练。使用feature_selection库的RFE类来选择特征的代码如下：
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数7 
new_x_train=RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(x_train,y_AQI_train)
#print(new_x_train)
#----------------------------------------------------------------------------
#树模型中GBDT也可用来作为基模型进行特征选择，使用feature_selection库的SelectFromModel类结合GBDT模型，来选择特征的代码如下：
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
#GBDT作为基模型的特征选择
#new_x_train=SelectFromModel(GradientBoostingClassifier()).fit_transform(x_train,y_AQI_train)
#print(new_x_train)
#------------------------------------------------------------------------------
#

##############################降维方法######################################
#------------------------------------------------------------------------
#使用decomposition库的PCA类选择特征的代码如下：(无监督的降维方式)
from sklearn.decomposition import PCA
#主成分分析法，返回降维后的数据4 
#参数n_components为主成分数目5 
new_x_train = PCA(n_components=3).fit_transform(x_train)
#print(new_x_train)
#-------------------------------------------------------------------------
#使用lda库的LDA类选择特征的代码如下：
from sklearn.lda import LDA
#线性判别分析法，返回降维后的数据4 #参数n_components为降维后的维数5 
new_x_train=LDA(n_components=3).fit_transform(x_train,y_AQI_train)
#print(new_x_train)

