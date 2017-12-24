import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV,Lasso
from sklearn.linear_model import RidgeCV ,Ridge
from sklearn.ensemble import  RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR


X_train=pd.read_csv('E:\\house\\x_train1.csv')
y_train=pd.read_csv('E:\\house\\y_train1.csv')
y_train.columns=['price']
X_test=pd.read_csv('E:\\house\\x_test1.csv')



'''
###################################################### feature selecting ####################################
top_fea=472
model = RandomForestRegressor(random_state=1)
model.fit(X_train, y_train)
feature_imp = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=["importance"])
feat_rf = feature_imp.sort_values("importance", ascending=False).head(top_fea).index
#reduce features
features = np.hstack([feat_rf])
features = list(np.unique(features))
#reduce features

X_train=X_train[features]

X_test=X_test[features]
'''

############################################# 但模型效果验证方法 ######################################
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
def rmse_cv(model, X_train, y_train):
	rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="mean_squared_error", cv=2))
	return(rmse)
import warnings
warnings.filterwarnings('ignore')
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error


#######################################  model ensemble  ####################################

## 1. bagging method ----------------------------------------------------

model=BaggingRegressor(
base_estimator=Lasso(alpha=.0005,max_iter=80),
n_estimators=1000, 
max_samples=0.8, 
max_features=0.8)
#print('bagging',rmse_cv(model, X_train, y_train).mean())

model=BaggingRegressor(
base_estimator=Lasso(alpha=.0005,max_iter=80),
n_estimators=100, 
max_samples=0.6, 
max_features=0.9,)
#print('bagging',rmse_cv(model, X_train, y_train).mean())


'''
# 2. adaboost method -----------------------------------------------
model=AdaBoostRegressor(base_estimator= Lasso(alpha=.0005,max_iter=80), n_estimators=8, learning_rate=0.01, loss='linear', random_state=None)
print('adaboost',rmse_cv(model, X_train, y_train).mean())
'''


#  3.  stacking --------------------------------------------------------

L1=list(range(0,291))
L2=list(range(291,582))
L3=list(range(582,873))
L4=list(range(873,1164))
L5=list(range(1164,1455))


list_index1=[L1+L2+L3+L4, L2+L3+L4+L5, L1+L5+L3+L4, L1+L2+L5+L4, L1+L2+L3+L5] #train
list_index2=[L5,L1,L2,L3,L4] #test

'''
L1=list(range(0,485))
L2=list(range(485,970))
L3=list(range(970,1455))
list_index1=[L1+L2, L2+L3, L1+L3, ] #train
list_index2=[L3,L1,L2] #test
'''
'''
L1=list(range(0,364))
L2=list(range(364,728))
L3=list(range(728,1092))
L4=list(range(1092,1456))
list_index1=[L4+L2+L3, L1+L3+L4, L1+L2+L4, L1+L2+L3] #train
list_index2=[L1,L2,L3,L4] #test
'''

'''
L1=list(range(0,729))
L2=list(range(729,1458))
list_index1=[L1, L2] #train
list_index2=[L2,L1] #test
'''

def stacking_train(x_train,y_train,list_index1,list_index2,x):

	k_fold=5
	k_fold_train_num=291 #k_fold 中每一层的样本数

	models=[
	LassoCV(),
	RidgeCV(),
	]


	m=len(models) #stacking构成训练集合的特征数

	model_second=Lasso(alpha=0.0005,max_iter=70) #XGBRegressor(n_estimators=7000,max_depth=6)


	n_train=len(x_train)#获取stacking构成训练集合的样本数
	data_stacking_train=pd.DataFrame(np.zeros((n_train,m)))

	data_stacking_submit=pd.DataFrame(np.zeros((len(x),m)))


	for i in list(range(k_fold)):#将各层训练集合，通过第一层的模型训练预测为stacking训练集合中生成特征元素
		
		pred=[]
		pred_test=[]
		pred_submit=[]

		index1=list_index1[i]
		index2=list_index2[i]

		for model in models: 

		
			model.fit(x_train.iloc[index1,:],y_train.iloc[index1,:])
			pred.append(model.predict(x_train.iloc[index2,:]))

			pred_submit.append(model.predict(x))


		for j in range(k_fold_train_num):#将第一层的预测值作为特征值赋值给新构造的stacking训练集合---------------------
			b=index2[0]
	
			
			data_stacking_train.iloc[j+b,0]=pred[0][j]
			data_stacking_train.iloc[j+b,1]=pred[1][j]
		
			
		
		for k in range(len(x)):#用第一层的模型来给submit集合生成特征元素-----------------------------------
		
			data_stacking_submit.iloc[k,0]+=pred_submit[0][k]
			data_stacking_submit.iloc[k,1]+=pred_submit[1][k]

		
		print(i)

	######### 利用data——stacking——train 进行第二层模型的训练
	model_2=model_second
	model_2.fit(data_stacking_train,y_train)

	return (model_2.predict(data_stacking_submit/k_fold))


# 4.average method -----------------------------------------------
models=[

Lasso(alpha=0.0004, max_iter=80),
Lasso(alpha=0.0004, max_iter=70),
Lasso(alpha=0.0005,max_iter=70),
Lasso(alpha=0.0005, max_iter=80),
Lasso(alpha=0.0005, max_iter=80),
Lasso(alpha=0.0006, max_iter=80),
Lasso(alpha=0.0006, max_iter=80),


LassoCV(max_iter=100),
LassoCV(max_iter=200),
LassoCV(max_iter=300),
LassoCV(max_iter=400),
LassoCV(max_iter=500),
LassoCV(max_iter=600),
LassoCV(max_iter=700),
LassoCV(max_iter=800),
LassoCV(max_iter=900),
LassoCV(max_iter=1000),

BaggingRegressor(
base_estimator=Lasso(alpha=.0005,max_iter=80),
n_estimators=50, 
max_samples=0.6, 
max_features=0.9,),

BaggingRegressor(
base_estimator=Lasso(alpha=.0005,max_iter=80),
n_estimators=100, 
max_samples=0.6, 
max_features=0.9,),

BaggingRegressor(
base_estimator=Lasso(alpha=.0005,max_iter=80),
n_estimators=150, 
max_samples=0.6, 
max_features=0.9,),

BaggingRegressor(
base_estimator=Lasso(alpha=.0005,max_iter=80),
n_estimators=200, 
max_samples=0.6, 
max_features=0.9,),

AdaBoostRegressor(base_estimator= Lasso(alpha=.0005,max_iter=80),
 n_estimators=8, 
 learning_rate=0.01, 
 loss='linear', 
 random_state=None),

AdaBoostRegressor(base_estimator= Lasso(alpha=.0005,max_iter=80),
 n_estimators=20, 
 learning_rate=0.01, 
 loss='linear', 
 random_state=None),

AdaBoostRegressor(base_estimator= Lasso(alpha=.0005,max_iter=80),
 n_estimators=40, 
 learning_rate=0.01, 
 loss='linear', 
 random_state=None),



BaggingRegressor(
base_estimator=Ridge(alpha=80),
n_estimators=50, 
max_samples=0.6, 
max_features=0.9,),

BaggingRegressor(
base_estimator=Ridge(alpha=80),
n_estimators=100, 
max_samples=0.6, 
max_features=0.9,),

BaggingRegressor(
base_estimator=Ridge(alpha=80),
n_estimators=150, 
max_samples=0.6, 
max_features=0.9,),

BaggingRegressor(
base_estimator=Ridge(alpha=80),
n_estimators=200, 
max_samples=0.6, 
max_features=0.9,),

AdaBoostRegressor(base_estimator= Ridge(alpha=80),
 n_estimators=8, 
 learning_rate=0.01, 
 loss='linear', 
 random_state=None),

AdaBoostRegressor(base_estimator=Ridge(alpha=80),
 n_estimators=20, 
 learning_rate=0.01, 
 loss='linear', 
 random_state=None),

AdaBoostRegressor(base_estimator= Ridge(alpha=80),
 n_estimators=40, 
 learning_rate=0.01, 
 loss='linear', 
 random_state=None)

]

'''
pred=np.array([0 for i in range(len(X_test))])# 相加起点

pred_=[]
for model in models:
	model.fit(X_train,y_train)
	pred_.append(model.predict(X_test))

for p in pred_:
	pred=pred+p/len(pred_)

'''

#################################### 提交submit ##########################################
# 给定一个 pred 数组作为提交

model=BaggingRegressor(
base_estimator=Lasso(alpha=.0005,max_iter=80),
n_estimators=1000, 
max_samples=0.7, 
max_features=0.9,)

model.fit(X_train,y_train)
pred=model.predict(X_test)


#pred=stacking_train(X_train,y_train,list_index1,list_index2,X_test)


y_test=(np.exp(list(pred)))
sns.distplot(y_test)
plt.show()


for i in range(len(y_test)):
	if y_test[i]>600000:
		print(i)
		y_test[i]=300000 # 发现预测的结果中有一个值疑似为异常值，900000左右，将其设置为200000

sub=[]
sample_submission=pd.read_csv('E:\\house\\sample_submission.csv')
sample_submission['SalePrice'] = pd.Series(y_test,index=sample_submission.index)

sample_submission.to_csv('E:\\house\\submission.csv',index=None)
