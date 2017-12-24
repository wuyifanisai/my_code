#用人工神经网络模型对多分类问题的机器学习，空气状况分类
#-*- coding: utf-8 -*-
def Sequential(data,classfy_data,data_test,classfy_data_test):
	from keras.models import Sequential
	from keras.layers.core import Dense,Activation,Dropout
	from keras.utils import np_utils
	import numpy as np

	uniques, ids = np.unique(classfy_data, return_inverse=True)
	label = np_utils.to_categorical(ids, len(uniques))
	###################################################################
	model = Sequential()
	model.add(Dense(output_dim=200, input_dim=6, activation='relu'))

	model.add(Dense(output_dim=100, input_dim=200, activation='relu'))

	model.add(Dense(output_dim=50, input_dim=100, activation='relu'))

	model.add(Dense(output_dim=20, input_dim=50, activation='relu'))

	model.add(Dense(output_dim=5, input_dim=20, activation='softmax'))

	model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

	'''
	model.fit(data, label, nb_epoch = 5000, batch_size = 20) 
	model.save_weights('e:\my_model_weights.h5')
	'''
########################################################################
	model.load_weights('E:\Master\PPDAMcode\AIR_project\my_model_weights.h5') 
	
	yp_test = model.predict_classes(data_test).reshape(len(data_test)) 
	k=0
	for i in range(len(data_test)):
		if (yp_test[i]+1)==classfy_data_test[i]:
			k=k+1
	print()
	print('correct rate of test is ',k/len(classfy_data_test),k,len(classfy_data_test))
