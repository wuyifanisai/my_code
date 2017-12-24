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
import warnings
warnings.filterwarnings("ignore")
'''
将构造出来的所有特征分为稀疏类以及稠密类特征，分别利用线性回归模型以及树结构模型，分析特征的重要程度

'''
X_train=pd.read_csv('E:\\house\\x_train.csv')
y_train=pd.read_csv('E:\\house\\y_train.csv')
y_train.columns=['price']
X_test=pd.read_csv('E:\\house\\x_test.csv')


sparse_feat=[]
dense_feat=[]

for feat in X_train.columns:
	if len(X_train[feat].value_counts())<10:
		sparse_feat.append(feat)
	else:
		dense_feat.append(feat)


print('sparse_feat',len(sparse_feat))
print('dense_feat',len(dense_feat))

X_train_sparse=X_train.loc[:,sparse_feat]
X_train_dense=X_train.loc[:,dense_feat]
print(X_train_sparse.shape)
print(X_train_dense.shape)


X_test_sparse=X_test.loc[:,sparse_feat]
X_test_dense=X_test.loc[:,dense_feat]
print(X_test_sparse.shape)
print(X_test_dense.shape)


##################################################### feat评估  ####################### #############################
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
def rmse_cv(model, X_train, y_train):
	rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="mean_squared_error", cv=5))
	return(rmse)
import warnings
warnings.filterwarnings('ignore')
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error


                     ###################### 针对sparse类型特征的评估 ####################

# Lasso ----------------------------------------------------------------------------------------------------------------------------------------
'''
alphas = [0.00001, 0.000035, 0.0004, 0.00047, 0.0005, 0.00053,0.0006, 0.007] #best 0.0005
cv_model = [rmse_cv(Lasso(alpha=alpha,max_iter=70), X_train_sparse, y_train).mean() for alpha in alphas]
result = pd.Series(cv_model, index = alphas)
result.plot()
print(result.min())
plt.show()

print('sparse feat importance by lasso:')
lasso=Lasso(alpha=0.0005,max_iter=70)
lasso.fit(X_train_sparse, y_train)
importance=list(lasso.coef_)

importance_feat_sparse_BYLASSO=[]#存储利用lasso筛选出来的重要的sparse类型的特征

for i in range(len(importance)):
	if importance[i] >0:
		importance_feat_sparse_BYLASSO.append(sparse_feat[i])
print('important feat of sparse by lasso:',importance_feat_sparse_BYLASSO)
print(len(importance_feat_sparse_BYLASSO))
'''



'''
lasso 建模时候，模型可以自动选择重要特征，所以不需要人为选择，人为事先选择反而会是模型效果降低
'''




'''
# gbm ------------------------------------------------------------------------------------------------------------------------------
num = [450,470,500,520,550] #470
depth_n=[1,2,3,4,5,6] #2
cv_model = [rmse_cv(GradientBoostingRegressor(n_estimators=150,max_depth=n), X_train_sparse, y_train).mean() for n in depth_n]

result = pd.Series(cv_model, index = depth_n)
result.plot()
print(result.min())
plt.show()

# 计算出每个sparse特征的重要度，利用gbm
gbrt=GradientBoostingRegressor(n_estimators=150,max_depth=3)
gbrt.fit(X_train_sparse, y_train)
import matplotlib.pyplot as plt
names=sparse_feat

# sort importances

print(gbrt.feature_importances_)

importance=list(gbrt.feature_importances_)
importance_feat_sparse_bygbm=[]#存储利用lasso筛选出来的重要的sparse类型的特征

for i in range(len(importance)):
	if importance[i] >1e-3:
		importance_feat_sparse_bygbm.append(sparse_feat[i])
print('important feat of sparse by gbm:',importance_feat_sparse_bygbm)
print(len(importance_feat_sparse_bygbm))
'''


'''
如果在用xgb建模时候，只利用lasso提取出来的 importance_feat_sparse，效果并没有提升, 利用gbm提取出来的 importance_feat_sparse，效果有微小提升
'''


 
                              ###################### 针对dense类型特征的评估 ####################
'''
# Lasso ----------------------------------------------------------------------------------------------------------------------------------------

alphas = [0.00001, 0.000035, 0.0004, 0.00047, 0.0005, 0.00053,0.0006, 0.007] #best 0.0005
cv_model = [rmse_cv(Lasso(alpha=alpha,max_iter=70), X_train_dense, y_train).mean() for alpha in alphas]
result = pd.Series(cv_model, index = alphas)
result.plot()
print(result.min())
plt.show()

print('sparse feat importance by lasso:')
lasso=Lasso(alpha=0.0005,max_iter=70)
lasso.fit(X_train_dense, y_train)
importance=list(lasso.coef_)

importance_feat_dense_BYLASSO=[]#存储利用lasso筛选出来的重要的sparse类型的特征

for i in range(len(importance)):
	if importance[i] >0:
		importance_feat_dense_BYLASSO.append(dense_feat[i])
print('important feat of dense by lasso:',importance_feat_dense_BYLASSO)
print(len(importance_feat_dense_BYLASSO))

'''
#lasso 建模时候，模型可以自动选择重要特征，所以不需要人为选择，人为事先选择反而会是模型效果降低
'''


# gbm ------------------------------------------------------------------------------------------------------------------------------
num = [450,470,500,520,550] #470
depth_n=[1,2,3,4,5,6] #2
cv_model = [rmse_cv(GradientBoostingRegressor(n_estimators=150,max_depth=n), X_train_dense, y_train).mean() for n in depth_n]

result = pd.Series(cv_model, index = depth_n)
result.plot()
print(result.min())
plt.show()
'''