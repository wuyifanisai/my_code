def DecisionTreeClassifier(data,classfy_data,data_test,classfy_data_test):
#use DecisionTreeClassifier
	from sklearn.tree import DecisionTreeClassifier as DTC

	dtc=DTC(criterion='entropy')
	dtc.fit(data,classfy_data)
	n=0
	for i in range(len(data_test)):
		info=data_test[i]
		s=dtc.predict(info)
		if s==classfy_data_test[i]:
			n=n+1
		else:
			pass
	print(' right rate of DecisionTreeClassifier  training is' ,float(n/len(data_test)))