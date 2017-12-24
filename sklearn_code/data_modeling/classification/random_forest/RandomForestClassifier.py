# use RFC model 
def RandomForestClassifier(data,classfy_data,data_test,classfy_data_test):
	from sklearn.ensemble import RandomForestClassifier
	rfc=RandomForestClassifier()
	rfc.fit(data,classfy_data)

###############test############
	print()
	predicted=rfc.predict(data_test)

	k=0
	for i in range(len(classfy_data_test)):
		if predicted[i]==classfy_data_test[i]:
			k=k+1
	print('RandomForestClassifier rate of test is :',k/len(data_test))


	k=0
	for i in range(len(classfy_data_test)):
		e=0
		e=abs(predicted[i]-classfy_data_test[i])/classfy_data_test[i]
		k=k+e
	print('RandomForestClassifier rate of test is :',1-k/len(data_test))

	#output the importance of feature
	import matplotlib.pyplot as plt
	f, ax = plt.subplots(figsize=(7, 5))
	ax.bar(range(len(rfc.feature_importances_)),rfc.feature_importances_)
	ax.set_title("Feature Importances")
	f.show()
	print('根据RandomForestClassifier得出特征重要程度')
	print(rfc.feature_importances_)


	f, ax = plt.subplots(figsize=(7, 5))
	ax.bar(range(len(rfc.feature_importances_)),rfc.feature_importances_)
	ax.set_title("Feature Importances")
	f.show()
	print('根据RandomForestClassifier得出特征重要程度')
	print(rfc.feature_importances_)