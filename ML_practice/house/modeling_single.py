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
from sklearn.neighbors import KNeighborsRegressor


X_train0=pd.read_csv('E:\\house\\x_train.csv')
X_test0=pd.read_csv('E:\\house\\x_test.csv')
y_train0=pd.read_csv('E:\\house\\y_train.csv')

X_train1=pd.read_csv('E:\\house\\x_train1.csv')
X_test1=pd.read_csv('E:\\house\\x_test1.csv')
y_train1=pd.read_csv('E:\\house\\y_train1.csv')

X_train0.drop([690,1181],axis=0,inplace=True)
y_train0.drop([690,1181],axis=0,inplace=True)
X_train0.index=range(len(X_train0))
y_train0.index=range(len(y_train0))


more_train=pd.concat((X_train0,X_train1),axis=1)
more_test=pd.concat((X_test0,X_test1),axis=1)

print(more_test.shape)
print(more_train.shape)


X_train=more_train
y_train=y_train0
X_test=more_test


###################################################### feature selecting ####################################
top_fea=450
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


############################################# 但模型效果验证方法 ######################################
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
def rmse_cv(model, X_train, y_train):
	rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="mean_squared_error", cv=3))
	return(rmse)
import warnings
warnings.filterwarnings('ignore')
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error



########################################### 单模型参数调优  ###############################################
'''
# KNN ------------------------------------------------
nn=[1,2,3,4,5,6,8,9,10] # best 6
cv_model = [rmse_cv(KNeighborsRegressor(n_neighbors=n, weights='uniform', algorithm='auto', leaf_size=20, p=2,)
, X_train, y_train).mean() for n in nn]
result = pd.Series(cv_model, index = nn)
result.plot()
print(result.min())
plt.title('knn with cs')
plt.show()
'''

'''
#POLY _LINEAR ------------------------------------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
model = Pipeline([('poly', PolynomialFeatures(degree=1)),('linear', LinearRegression(fit_intercept=False))])
print(rmse_cv(model, X_train, y_train).mean())
'''


#SVM--------------------------------------------
cs = [0.17,0.18,0.2,0.25,0.3] #best 0.0005
cv_model = [rmse_cv(LinearSVR(C=c), X_train, y_train).mean() for c in cs]
result = pd.Series(cv_model, index = cs)
result.plot()
print(result.min())
plt.title('svm with cs')
plt.show()



# Lasso -----------------------------------------
alphas = [0.0002, 0.0003, 0.0004, 0.00047, 0.0005, 0.00055,0.0006, 0.0007] #best 0.0005
alphas = [0.0001,0.0003,0.0004,0.0005,0.0006,0.0008,0.001,0.0012,0.0013,0.0015,0.0017,0.002]
cv_model = [rmse_cv(Lasso(alpha=alpha,max_iter=70), more_train, y_train0).mean() for alpha in alphas]
result = pd.Series(cv_model, index = alphas)
result.plot()
print(result.min())
plt.title('lasso with alphas')
plt.show()


'''
# ridge ------------------------------------------ 
alphas = [10,20,30,40,50,60,70,80,90,100,110,130] # 80
cv_model = [rmse_cv(Ridge(alpha=n), more_train, y_train0).mean() for n in alphas]
result = pd.Series(cv_model, index = alphas)
print(result.min())
result.plot()
plt.title('ridge with alphas')
plt.show()
'''

'''
# DNN -----------------------------------------------
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.utils import np_utils
model = Sequential()
model.add(Dense(output_dim=1000, input_dim=len(X_train.columns), activation='relu'))
model.add(Dense(output_dim=200, input_dim=1000, activation='relu'))
model.add(Dense(output_dim=1, input_dim=200, activation='relu'))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(X_train.as_matrix(), y_train.as_matrix(), nb_epoch = 200, batch_size = 100) 
pred=model.predict(X_test.as_matrix()).reshape(len(X_test)) 
'''



'''
# gbrt -----------------------------------------
num = [150,200,250]   #best 200
depth_n=[2,10,20,50]  #2

cv_model = [rmse_cv(GradientBoostingRegressor(n_estimators=200,max_depth=n), more_train, y_train0).mean() for n in depth_n]
result = pd.Series(cv_model, index = depth_n)
print(result.min())
result.plot()
plt.show()
'''


'''
#rf -------------------------------------------
num = [10,50,70,100,120,150,200] #70
depth_n=[100] #100
cv_model = [rmse_cv(RandomForestRegressor(n_estimators=70,max_depth=n), X_train, y_train).mean() for n in depth_n]
result = pd.Series(cv_model, index = depth_n)
result.plot()
print(result.min())
plt.show()
'''

'''
# xgb --------------------------------------
num = [400,500,600] #500
depth_n=[1,2,3,4,5,6] #3
cv_model = [rmse_cv(XGBRegressor(n_estimators=n,max_depth=3), X_train, y_train).mean() for n in num]
result = pd.Series(cv_model, index = num)
result.plot()
plt.show()
'''



############################################# 单模型调优后效果对比  ############################################################
print('lasso:',rmse_cv(Lasso(alpha=0.0005,max_iter=70), X_train, y_train).mean())
print('gbrt',rmse_cv(GradientBoostingRegressor(n_estimators=200,max_depth=2), X_train, y_train).mean())
print('rf',rmse_cv(RandomForestRegressor(n_estimators=70,max_depth=100), X_train, y_train).mean())
print('xgb',rmse_cv(XGBRegressor(n_estimators=500,max_depth=3), X_train, y_train).mean())
print('lassocv',rmse_cv(LassoCV(), X_train, y_train).mean())
print('ridgecv',rmse_cv(RidgeCV(), X_train, y_train).mean())

'''
lasso: 0.110979658332
gbrt 0.123508522367
rf 0.134636833298
xgb 0.120726773952
lassocv 0.110801115946
ridgecv 0.112586018813

'''

#################################### 提交submit ##########################################
# 给定一个 pred 数组作为提交
'''
model=Lasso(alpha=0.0005,max_iter=70)
model.fit(X_train, y_train)
pred=model.predict(X_test)
'''

y_test=(np.exp(list(pred)))
sns.distplot(y_test)
plt.show()

sub=[]
sample_submission=pd.read_csv('E:\\house\\sample_submission.csv')
sample_submission['SalePrice'] = pd.Series(y_test,index=sample_submission.index)
print(sample_submission)
sample_submission.to_csv('E:\\house\\submission.csv',index=None)
