#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

gbrt=GradientBoostingRegressor(
  loss='ls'
, learning_rate=0.2
, n_estimators=2 #number of regression trees
, subsample=1
, min_samples_split=2
, min_samples_leaf=1
, max_depth=2 #depth of each individual tree
, init=None
, random_state=None
, max_features=None
, alpha=0.9
, verbose=0
, max_leaf_nodes=None
, warm_start=False
)


gbdt=GradientBoostingClassifier(
  loss='deviance'
, learning_rate=0.2
, n_estimators=2 #number of regression trees
, subsample=1
, min_samples_split=2
, min_samples_leaf=1
, max_depth=15 #depth of each individual tree
, init=None
, random_state=None
, max_features=None
, verbose=0
, max_leaf_nodes=None
, warm_start=False 
)




all_data=pd.read_excel('E:\Master\PPDAMcode\AIR_project\hangzhou_air_alldata01.xls')
names=[1,2,3]
import random
index=list(range(1000))
test_index = random.sample(index, 800)##test_index is the index of test data随机选出200个样本作为测试样本
train_index=[]
for i in range(1000):
	if i not in test_index:
		train_index.append(i)   ##train_index is the index of train data

x_train=all_data.iloc[train_index,0:3].as_matrix()
y_train=all_data.iloc[train_index,6].as_matrix()

x_test=all_data.iloc[test_index,0:3].as_matrix()
y_test=all_data.iloc[test_index,6].as_matrix()

x_AQI_train=all_data.iloc[train_index,7].as_matrix()
y_AQI_test=all_data.iloc[test_index,7].as_matrix()


gbrt.fit(x_train,y_train)


pred=gbrt.predict(x_test)

k=0
for i in range(len(pred)):
	k=k+abs(pred[i]-y_test[i])/(y_test[i])
print(1-k/len(pred))




# sort importances
indices = np.argsort(gbrt.feature_importances_)
# plot as bar chart
plt.barh(np.arange(len(names)), gbrt.feature_importances_[indices])
plt.yticks(np.arange(len(names)) + 0.25, np.array(names)[indices])
_ = plt.xlabel('Relative importance')
#plt.show()


print(gbrt.feature_importances_[indices])


#print(gbdt.score(x_test,y_test))  # score on test data (accuracy)
print('###################################')
print(gbrt.apply(np.array(x_test)))
for i in range(len(pred)):
	print(pred[i],gbrt.apply(np.array(x_test))[i][0])

'''
for i in range(len(y_train)):
	print(gbdt.fit_transform(x_train,y_train)[i],y_train[i])
'''

#初步结论：apply()函数可以将样本落在哪个叶子节点处的位置用向量表示出，构成一个新的特征
#fit-TRANSFORM()函数可以将训练样本的特征进行转换，起到类似降为的作用