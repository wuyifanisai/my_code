import pandas as pd
import numpy as np
from numpy import matrix
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from IPython.display import Image

data=pd.read_csv('E:\\kaggle_titanic\\clean_train.csv')
data_test=pd.read_csv("e:\\kaggle_titanic\\clean_test.csv")


############################################## 构造预测是否生还的DATA ###################################################

x=data.iloc[:,[1,2,3,4,5,6,7,8,9]].as_matrix()
y=data.iloc[:,0].as_matrix()
#3-fold stacking 

#prepare base model--------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression as LR
lr=LR(
penalty='l2', 
dual=False, 
tol=0.001, 
C=100.0, 
fit_intercept=True, 
intercept_scaling=1, 
class_weight=None,
random_state=None, 
solver='liblinear', 
max_iter=100, 
multi_class='ovr',
verbose=0, 
warm_start=False,
n_jobs=1)
#-----------------------------------------------------------------------------------------
from sklearn import svm
svm = svm.SVC(
C=1.0, 
kernel='rbf',
degree=3, 
gamma='auto', 
coef0=0.0, 
shrinking=True, 
probability=False, 
tol=0.001, 
cache_size=200, 
class_weight=None, 
verbose=False, 
max_iter=1000, 
decision_function_shape=None, 
random_state=None)
#----------------------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(
criterion='entropy',
splitter='best',
max_depth=4,
min_samples_split=2, 
min_samples_leaf=1,
min_weight_fraction_leaf=0.0, 
max_features=None, 
random_state=None,
max_leaf_nodes=None, 
min_impurity_split=1e-07, 
class_weight=None, 
presort=False)
#--------------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(
n_estimators=100, 
	#criterion=None,
	max_depth=5,
	min_samples_split=2,
	min_samples_leaf=1,
	min_weight_fraction_leaf=0.0, 
	max_features=1.0, 
	max_leaf_nodes=None, 
	min_impurity_split=1e-07,
	bootstrap=True, 
	oob_score=False, 
	n_jobs=1, 
	random_state=None, 
	verbose=0, 
	warm_start=False)
#----------------------------------------------------------------------------------------------
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.utils import np_utils
model = Sequential()
model.add(Dense(output_dim=20, input_dim=9, activation='relu'))
model.add(Dense(output_dim=1, input_dim=20, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
#--------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
x_stacking=pd.DataFrame(np.zeros((len(x),5)))
x_test_stacking=pd.DataFrame(np.zeros((len(data_test),5)))

################################## fold-1 ######################################################################
lr.fit(x[0:593],y[0:593])
svm.fit(x[0:593],y[0:593])
dtc.fit(x[0:593],y[0:593])
rf.fit(x[0:593],y[0:593])
model.fit(x[0:593],y[0:593], nb_epoch = 50, batch_size = 10) 

#--------------------------------------------------
pred_lr=lr.predict(x[593:889])
pred_svm=svm.predict(x[593:889])
pred_dtc=dtc.predict(x[593:889])
pred_rf=rf.predict(x[593:889])
pred_model=model.predict_classes(x[593:889])

for i in range(296):
	x_stacking.iloc[i+593,0]=pred_lr[i]
	x_stacking.iloc[i+593,1]=pred_svm[i]
	x_stacking.iloc[i+593,2]=pred_dtc[i]
	x_stacking.iloc[i+593,3]=pred_rf[i]
	x_stacking.iloc[i+593,4]=pred_model[i]
#---------------------------------------------------



#----------------------------------------------------
pred_lr=lr.predict(data_test)
pred_svm=svm.predict(data_test)
pred_dtc=dtc.predict(data_test)
pred_rf=rf.predict(data_test)
pred_model=model.predict_classes(data_test.as_matrix())

for i in range(len(data_test)):
	x_test_stacking.iloc[i,0]=x_test_stacking.iloc[i,0]+pred_lr[i]
	x_test_stacking.iloc[i,1]=x_test_stacking.iloc[i,1]+pred_svm[i]
	x_test_stacking.iloc[i,2]=x_test_stacking.iloc[i,2]+pred_dtc[i]
	x_test_stacking.iloc[i,3]=x_test_stacking.iloc[i,3]+pred_rf[i]
	x_test_stacking.iloc[i,4]=x_test_stacking.iloc[i,4]+pred_model[i]
#---------------------------------------------------


############################ fold-2 ############################################################################
model = Sequential()
model.add(Dense(output_dim=20, input_dim=9, activation='relu'))
model.add(Dense(output_dim=1, input_dim=20, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

lr.fit(x[296:889],y[296:889])
svm.fit(x[296:889],y[296:889])
dtc.fit(x[296:889],y[296:889])
rf.fit(x[296:889],y[296:889])
model.fit(x[296:889],y[296:889], nb_epoch = 50, batch_size = 10) 

#---------------------------------------
pred_lr=lr.predict(x[0:296])
pred_svm=svm.predict(x[0:296])
pred_dtc=dtc.predict(x[0:296])
pred_rf=rf.predict(x[0:296])
pred_model=model.predict_classes(x[0:296])

for i in range(296):
	x_stacking.iloc[i,0]=pred_lr[i]
	x_stacking.iloc[i,1]=pred_svm[i]
	x_stacking.iloc[i,2]=pred_dtc[i]
	x_stacking.iloc[i,3]=pred_rf[i]
	x_stacking.iloc[i,4]=pred_model[i]
#---------------------------------------

#-----------------------------------------------
pred_lr=lr.predict(data_test)
pred_svm=svm.predict(data_test)
pred_dtc=dtc.predict(data_test)
pred_rf=rf.predict(data_test)
pred_model=model.predict_classes(data_test.as_matrix())

for i in range(len(data_test)):
	x_test_stacking.iloc[i,0]=x_test_stacking.iloc[i,0]+pred_lr[i]
	x_test_stacking.iloc[i,1]=x_test_stacking.iloc[i,1]+pred_svm[i]
	x_test_stacking.iloc[i,2]=x_test_stacking.iloc[i,2]+pred_dtc[i]
	x_test_stacking.iloc[i,3]=x_test_stacking.iloc[i,3]+pred_rf[i]
	x_test_stacking.iloc[i,4]=x_test_stacking.iloc[i,4]+pred_model[i]
#------------------------------------

########################### fold 3 ###################################################
model = Sequential()
model.add(Dense(output_dim=20, input_dim=9, activation='relu'))
model.add(Dense(output_dim=1, input_dim=20, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

lr.fit(x[list(range(0,296))+list(range(593,889))],y[list(range(0,296))+list(range(593,889))])
svm.fit(x[list(range(0,296))+list(range(593,889))],y[list(range(0,296))+list(range(593,889))])
dtc.fit(x[list(range(0,296))+list(range(593,889))],y[list(range(0,296))+list(range(593,889))])
rf.fit(x[list(range(0,296))+list(range(593,889))],y[list(range(0,296))+list(range(593,889))])
model.fit(x[list(range(0,296))+list(range(593,889))],y[list(range(0,296))+list(range(593,889))], nb_epoch = 50, batch_size = 10) 

#-------------------------------------------------------------
pred_lr=lr.predict(x[296:593])
pred_svm=svm.predict(x[296:593])
pred_dtc=dtc.predict(x[296:593])
pred_rf=rf.predict(x[296:593])
pred_model=model.predict_classes(x[296:593])

for i in range(296):
	x_stacking.iloc[i+296,0]=pred_lr[i]
	x_stacking.iloc[i+296,1]=pred_svm[i]
	x_stacking.iloc[i+296,2]=pred_dtc[i]
	x_stacking.iloc[i+296,3]=pred_rf[i]
	x_stacking.iloc[i+296,4]=pred_model[i]
#----------------------------------------------------------

#-------------------------------------------------------------
pred_lr=lr.predict(data_test)
pred_svm=svm.predict(data_test)
pred_dtc=dtc.predict(data_test)
pred_rf=rf.predict(data_test)
pred_model=model.predict_classes(data_test.as_matrix())

for i in range(len(data_test)):
	x_test_stacking.iloc[i,0]=x_test_stacking.iloc[i,0]+pred_lr[i]
	x_test_stacking.iloc[i,1]=x_test_stacking.iloc[i,1]+pred_svm[i]
	x_test_stacking.iloc[i,2]=x_test_stacking.iloc[i,2]+pred_dtc[i]
	x_test_stacking.iloc[i,3]=x_test_stacking.iloc[i,3]+pred_rf[i]
	x_test_stacking.iloc[i,4]=x_test_stacking.iloc[i,4]+pred_model[i]
#--------------------------------------------------------------

print(x_test_stacking)

for i in range(len(x_test_stacking)):
	for j in range(5):
		if x_test_stacking.iloc[i,j]<2.5:
			x_test_stacking.iloc[i,j]=0.0
		else:
			x_test_stacking.iloc[i,j]=1.0
print(x_test_stacking)

####################################train the second model ##########################################
#第二层用gbdt

'''
from sklearn.ensemble import GradientBoostingClassifier
gbdt=GradientBoostingClassifier(
  loss='deviance'
, learning_rate=0.3
, n_estimators=25 #number of regression trees
, subsample=1.0
, min_samples_split=2
, min_samples_leaf=1
, max_depth=3 #depth of each individual tree
, init=None
, random_state=None
, max_features=1.0
, verbose=0
, max_leaf_nodes=None
, warm_start=True 
)	
gbdt.fit(x_stacking,y)

pred=gbdt.predict(x_test_stacking)

print(pred)
sample=pd.read_csv('E:\\kaggle_titanic\\gender_submission.csv')
for i in range(len(sample)):
	sample.iloc[i,1]=pred[i]
#print(sample)
sample.to_csv('E:\\kaggle_titanic\\stacking_submission.csv')
'''


#第二层用xgboost
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pyplot as plt
def modelfit(alg,  predictors,target,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
	if useTrainCV:
		xgb_param = alg.get_xgb_params()

		xgtrain = xgb.DMatrix(predictors, target)

		cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)

		alg.set_params(n_estimators=cvresult.shape[0])
		#Fit the algorithm on the data
		alg.fit(predictors, target,eval_metric='auc')

		#Predict training set:
		dtrain_predictions = alg.predict(predictors)
		dtrain_predprob = alg.predict_proba(predictors)[:,1]

		#Print model report:
		print ("\nModel Report")
		print ("Accuracy : %.4g" % metrics.accuracy_score(target, dtrain_predictions))
		print ("AUC Score (Train): %f" % metrics.roc_auc_score(target, dtrain_predprob))

		feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
		feat_imp.plot(kind='bar', title='Feature Importances')
		plt.ylabel('Feature Importance Score')
		plt.show()


data = pd.read_csv('E:\\kaggle_titanic\\clean_train.csv')

#Choose all predictors except target & IDcols
predictors=data.iloc[:,[1,2,3,4,5,6,7,8,9]]
target=data.iloc[:,0]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, x_stacking,y)


data=pd.read_csv("e:\\kaggle_titanic\\clean_test.csv")
pred=xgb1.predict(x_test_stacking)

sample=pd.read_csv('E:\\kaggle_titanic\\gender_submission.csv')
for i in range(len(sample)):
	sample.iloc[i,1]=pred[i]
#print(sample)
sample.to_csv('E:\\kaggle_titanic\\xgboost_submission.csv')