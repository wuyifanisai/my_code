#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier,XGBRegressor
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
modelfit(xgb1,predictors,target)

data=pd.read_csv("e:\\kaggle_titanic\\clean_test.csv")
pred=xgb1.predict(data)
sample=pd.read_csv('E:\\kaggle_titanic\\gender_submission.csv')
for i in range(len(sample)):
	sample.iloc[i,1]=pred[i]
print(sample)
#sample.to_csv('E:\\kaggle_titanic\\xgboost_submission.csv')

