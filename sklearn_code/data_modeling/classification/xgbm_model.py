#################################################################################################
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pyplot as plt

def modelfit_gbm(alg, predictors, target,performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(predictors, target)

    #Predict training set:
    dtrain_predictions = alg.predict(predictors)
    dtrain_predprob = alg.predict_proba(predictors)[:,1]

    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg,predictors, target, cv=cv_folds, scoring='roc_auc')

    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(target, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(target, dtrain_predprob))

    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors.columns).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        #plt.show()

#Choose all predictors except target & IDcols
data = pd.read_csv('E:\\kaggle_titanic\\clean_train.csv')
predictors=data.iloc[:,[1,2,3,4,5,6,7,8,9]]
target=data.iloc[:,0]

gbdt_0=GradientBoostingClassifier(
  loss='deviance'
, learning_rate=0.4
, n_estimators=5 #number of regression trees
, subsample=1
, min_samples_split=2
, min_samples_leaf=1
, max_depth=3 #depth of each individual tree
, init=None
, random_state=None
, max_features=None
, verbose=0
, max_leaf_nodes=None
, warm_start=False 
)   
modelfit_gbm(gbdt_0, predictors,target)

###################################################################################################
