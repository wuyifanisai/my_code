import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LassoCV,Lasso
from sklearn.linear_model import RidgeCV,Ridge
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import BaggingRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor


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


x_train=train.drop('y',axis=1)
x_test=test




############################################# 模型效果验证方法 ######################################
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
def rmse_cv(model, X_train, y_train):
	rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="mean_squared_error", cv=3))
	return(rmse)
import warnings
warnings.filterwarnings('ignore')
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error


# Lasso -----------------------------------------

n = [0.01,0.015,0.02,0.025,0.03,0.035]
cv_model = [rmse_cv(Lasso(alpha=num,max_iter=1000),
x_train, y_train).mean() for num in n]

result = pd.Series(cv_model, index = n)
result.plot()
print(result.min())
plt.title('lasso with alphas')
plt.show()

model=Lasso(alpha=0.025)
model.fit(x_train,y_train)

pred=model.predict(x_test)


output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': pred})
output.to_csv('E:\\lasso.csv', index=False)



# XGBRegressor --------------------------------------
'''
num = [1000]
cv_model = [rmse_cv(XGBRegressor(max_depth=4, learning_rate=0.005, n_estimators=n,
                    silent=True, objective="reg:linear",
                    gamma=0, min_child_weight=1,
                    subsample=1, colsample_bytree=1,
                    base_score=y_mean, seed=0, missing=None),
            x_train, y_train).mean() for n in num]

result = pd.Series(cv_model, index = num)
result.plot()
print(result.min())
plt.title(' XGBRegressor  ')
plt.show()
'''

'''
model=XGBRegressor(max_depth=4, learning_rate=0.005, n_estimators=850,
                    silent=True, objective="reg:linear",
                    subsample=0.93, 
                    base_score=y_mean, )

model.fit(x_train,y_train)
pred=model.predict(x_test)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': pred})
output.to_csv('E:\\XGBRegressor.csv', index=False)
'''

#######################################  model ensemble  ####################################

## 1. bagging method ----------------------------------------------------
'''
model=BaggingRegressor(
base_estimator=XGBRegressor(max_depth=4, learning_rate=0.005, n_estimators=800,
                    silent=True, objective="reg:linear",
                    subsample=0.95, 
                    base_score=y_mean, ),
n_estimators=10, 
max_samples=0.95, 
max_features=0.9
)

model.fit(x_train,y_train)
pred=model.predict(x_test)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': pred})
output.to_csv('E:\\bagging-XGBRegressor.csv', index=False)
'''
'''
## 2. adaboost method ----------------------------------------------------
model=AdaBoostRegressor(base_estimator=
 XGBRegressor(max_depth=4, learning_rate=0.005, n_estimators=800,
                    silent=True, objective="reg:linear",
                    subsample=0.95, 
                    base_score=y_mean,),
 n_estimators=10,
  learning_rate=0.01, 
  loss='linear', 
  random_state=None)

model.fit(x_train,y_train)
pred=model.predict(x_test)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': pred})
output.to_csv('E:\\adaboost-XGBRegressor.csv', index=False)
'''