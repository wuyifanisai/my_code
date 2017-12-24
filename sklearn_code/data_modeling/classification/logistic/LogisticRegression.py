#use LogisticRegression 
def LogisticRegression(data,classfy_data,data_test,classfy_data_test):
	from sklearn.linear_model import LogisticRegression as LR
	
	lr=LR()
	lr.fit(data,classfy_data)
	
	n=0

	for i in range(len(data_test)):
		if lr.predict(data_test[i])==classfy_data_test[i]:
			n=n+1
	print(' right rate of LogisticRegression  is ',n/len(data_test))
