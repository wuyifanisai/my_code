def SVM(data,classfy_data,data_test,classfy_data_test):
	import pandas as pd
	from numpy.random import shuffle #引入随机函数
	
#构造特征和标签
	data = data*30
	classfy_data = classfy_data.astype(int)
	data_test = data_test*30
	classfy_data_test = classfy_data_test.astype(int)

#导入模型相关的函数，建立并且训练模型
	from sklearn import svm
	model = svm.SVC()
	model.fit(data, classfy_data)

	y=model.predict(data_test)
	k=0
	for i in range(len(data_test)):
		if y[i]==classfy_data_test[i]:
			k=k+1
	print()
	print('the correct rate of SVM is ',(k/len(data_test)))
	print()

