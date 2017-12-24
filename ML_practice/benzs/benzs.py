import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LassoCV,Lasso
from sklearn.linear_model import RidgeCV ,Ridge
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import BaggingRegressor



train = pd.read_csv('E:\\benzs\\train.csv')
test = pd.read_csv('E:\\benzs\\test.csv')

#train=train[train.y<200]
# process columns, apply LabelEncoder to categorical features
# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))



from sklearn.decomposition import PCA, FastICA
n_comp = 10

# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]
    
y_train = train["y"]
y_mean = np.mean(y_train)

############################################### modeling #########################################
'''
model1=XGBRegressor(n_estimators=500,max_depth=4)
model1.fit(x_train,y_train)
pred=model1.predict(x_test)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': pred})
output.to_csv('E:\\xgboost.csv', index=False)


model2=BaggingRegressor(
base_estimator=XGBRegressor(n_estimators=500,max_depth=4),
n_estimators=100, 
max_samples=0.6, 
max_features=0.8)

model2.fit(x_train,y_train)
pred=model2.predict(x_test)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': pred})
output.to_csv('E:\\xgboost_bagging.csv', index=False)

'''
y_mean = np.mean(y_train)

import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_params = {
    'eta': 0.004,
    #'gamma':0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'verbose_eval':10,
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)
'''
# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=500, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)
'''
# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=900)


y_pred = model.predict(dtest)
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('E:\\xgboost-depth------{}-pca-ica.csv'.format(xgb_params['max_depth']), index=False)