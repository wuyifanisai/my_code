   #-*- coding: utf-8 -*-
def main():
	import read_data
	import data_standard
	import numpy as np
	import pandas as pd
	'''
	##############################################################数据读入与整理##################################################################################

	path2014='E:\Master\PPDAMcode\AIR_project\hangzhou.xls'
	path2015='E:\Master\PPDAMcode\AIR_project\hangzhou2015.xls'
	path2016='E:\Master\PPDAMcode\AIR_project\hangzhou2016.xls'

	hangzhou_data=[]
	##################################################################################
	data=read_data.read_data_xls(path2014)
	for i in range(len(data)-1):
		if data.iloc[i+1,7]=='优':
			data.iloc[i+1,7]=1.0
		elif data.iloc[i+1,7]=='良好':
			data.iloc[i+1,7]=2.0
		elif data.iloc[i+1,7]=='轻度污染':
			data.iloc[i+1,7]=3.0
		elif data.iloc[i+1,7]=='中度污染':
			data.iloc[i+1,7]=4.0
		elif data.iloc[i+1,7]=='重度污染':
			data.iloc[i+1,7]=5.0
	#for i in range(6):
		#data.iloc[1:,1+i]=data_standard.data_standard(data.iloc[1:,1+i])

	hangzhou_data.append(data)
	#################################################################################
	data=read_data.read_data_xls(path2015)
	for i in range(len(data)-1):
		if data.iloc[i+1,7]=='优':
			data.iloc[i+1,7]=1.0
		elif data.iloc[i+1,7]=='良好':
			data.iloc[i+1,7]=2.0
		elif data.iloc[i+1,7]=='轻度污染':
			data.iloc[i+1,7]=3.0
		elif data.iloc[i+1,7]=='中度污染':
			data.iloc[i+1,7]=4.0
		elif data.iloc[i+1,7]=='重度污染':
			data.iloc[i+1,7]=5.0
	#for i in range(6):
		#data.iloc[1:,1+i]=data_standard.data_standard(data.iloc[1:,1+i])

	hangzhou_data.append(data)
	##################################################################################
	data=read_data.read_data_xls(path2016)
	for i in range(len(data)-1):
		if data.iloc[i+1,7]=='优':
			data.iloc[i+1,7]=1.0
		elif data.iloc[i+1,7]=='良好':
			data.iloc[i+1,7]=2.0
		elif data.iloc[i+1,7]=='轻度污染':
			data.iloc[i+1,7]=3.0
		elif data.iloc[i+1,7]=='中度污染':
			data.iloc[i+1,7]=4.0
		elif data.iloc[i+1,7]=='重度污染':
			data.iloc[i+1,7]=5.0

	#for i in range(6):
		#data.iloc[1:,1+i]=data_standard.data_standard(data.iloc[1:,1+i])

	hangzhou_data.append(data)

	############################################################################
	#print(hangzhou_data)#######2014,2015,2016的所有数据在列表hangzhou_data中存放
	all_data=pd.DataFrame(np.linspace(1,1091*8,1091*8).reshape(1091,8))

	for i in range(len(hangzhou_data[0])-1):
		for j in range(8):
			all_data.iloc[i,j]=hangzhou_data[0].iloc[i+1,j+1]
			############################################
	for i in range(len(hangzhou_data[1])-1):
		for j in range(8):
			all_data.iloc[i+365,j]=hangzhou_data[1].iloc[i+1,j+1]
			###################################################
	for i in range(len(hangzhou_data[2])-1):
		for j in range(8):
			all_data.iloc[i+730,j]=hangzhou_data[2].iloc[i+1,j+1]

	#print(all_data)
	all_data.to_excel('E:\Master\PPDAMcode\AIR_project\hangzhou_air_alldata.xls')
	################################################################################################################################################

	'''
	############################主成分分析PCA######################################################
	import PCA
	PCA.PCA(all_data.iloc[:,0:6])

	#######################RandomizedLogisticRegression筛选特征##########
	from sklearn.linear_model import RandomizedLogisticRegression as RLR 
	rlr=RLR()
	rlr.fit(all_data.iloc[:,0:6].as_matrix(),all_data.iloc[:,6].as_matrix())
	print()
	print('通过RandomizedLogisticRegression筛选特征：')
	print(rlr.get_support())
	print()
	###############################################################################################



	#############################准备好训练数据与测试数据################################################################
	
	all_data=read_data.read_data_xls('E:\Master\PPDAMcode\AIR_project\yuanshi_data.xls')

	import random
	index=list(range(1091))
	test_index = random.sample(index, 90)##test_index is the index of test data随机选出200个样本作为测试样本
	train_index=[]
	for i in range(1091):
		if i not in test_index:
			train_index.append(i)   ##train_index is the index of train data

	data_train=all_data.iloc[train_index,0:6].as_matrix()
	classfy_train=all_data.iloc[train_index,6].as_matrix()
	AQI_train=all_data.iloc[train_index,7].as_matrix()
	

	data_test=all_data.iloc[test_index,0:6].as_matrix()
	classfy_test=all_data.iloc[test_index,6].as_matrix()
	AQI_test=all_data.iloc[test_index,7].as_matrix()

	########################################################################################################################

	'''
	############################运用DecisionTreeClassifier方法进行训练测试###############################
	import DecisionTreeClassifier
	DecisionTreeClassifier.DecisionTreeClassifier(data_train,AQI_train,data_test,AQI_test)
	######################################################################################################
	'''
	
	############################运用RandomForestClassifier方法进行训练测试###############################
	import RandomForestClassifier
	RandomForestClassifier.RandomForestClassifier(data_train,AQI_train,data_test,AQI_test)
	######################################################################################################
	'''
	############################运用SVM方法进行训练测试###############################
	import SVM
	SVM.SVM(data_train,classfy_train,data_test,classfy_test)	
	######################################################################################################
	'''
	'''
	############################运用keras方法进行训练测试###############################
	import Sequential
	Sequential.Sequential(data_train,classfy_train,data_test,classfy_test)
	######################################################################################################
	'''

	'''
	############################运用LogisticRegression方法进行训练测试###############################
	import LogisticRegression
	LogisticRegression.LogisticRegression(data_train,classfy_train,data_test,classfy_test)
	#################################################################################################
	'''



main()