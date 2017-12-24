import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LassoCV,Lasso
from sklearn.linear_model import RidgeCV ,Ridge
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import  RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
X = np.zeros((4,5))
print(X)
kf = KFold(n_splits=4)
for train, test in kf.split(X):
	print("%s %s" % (X[train], X[test]))

c=1/0
train = pd.read_csv('E:\\benzs\\train.csv')
test = pd.read_csv('E:\\benzs\\test.csv')
test0=test
# drop the unnormal point
train=train[train.y<200]


##################### one hot ##################################
test0=test

y_train=train['y']
train=train.drop(['ID','y'],axis=1)
train['train']=1
test=test.drop('ID',axis=1)
test['train']=0
print(train.shape)
print(test.shape)

#concat train and test
x=pd.concat((train,test),axis=0)
print(x.shape)

# one_hot object feature
obj_fea=list(x.dtypes[x.dtypes=='object'].index)
print(obj_fea)


one_hot_x=pd.get_dummies(x.loc[:,obj_fea])
one_hot_x.index=x.index
print(one_hot_x)
x=x.drop(obj_fea,axis=1)
x=pd.concat(( x,one_hot_x ),axis=1)

print(x.shape)

### get train and test
x_train=x[x.train==1]
x_test=x[x.train==0]


'''

##########################################  process columns, apply LabelEncoder to categorical features  ###################

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))
'''

'''
###################################################### feature selecting ##################################################

top_fea=350
model = RandomForestRegressor()
model.fit(x_train, y_train)
feature_imp = pd.DataFrame(model.feature_importances_, index=x_train.columns, columns=["importance"])
feat_rf = feature_imp.sort_values("importance", ascending=False).head(top_fea).index

names=list(x_train.columns)
# sort importances
indices = np.argsort(model.feature_importances_)
# plot as bar chart
plt.barh(np.arange(len(names)), model.feature_importances_[indices])
plt.yticks(np.arange(len(names)) + 0.25, np.array(names)[indices])
_ = plt.xlabel('Relative importance')
#plt.show()

#reduce features
features = np.hstack([feat_rf])
features = list(np.unique(features))
#reduce features

x_train=x_train[features]
x_test=x_test[features]
'''

############################################ make new features  ##############################################
from sklearn.decomposition import PCA, FastICA
n_comp = 10

# PCA
pca = PCA(n_components=n_comp)
pca2_results_train = pca.fit_transform(x_train)
pca2_results_test = pca.transform(x_test)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(x_train)
ica2_results_test = ica.transform(x_test)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    x_train['pca_' + str(i)] = pca2_results_train[:,i-1]
    x_test['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    x_train['ica_' + str(i)] = ica2_results_train[:,i-1]
    x_test['ica_' + str(i)] = ica2_results_test[:, i-1]
    
#y_train=train['y']
#x_train=train.drop(['y','ID'],axis=1)
#x_test=test.drop(['ID'],axis=1)


#x_train=x_train.iloc[:,-2*n_comp:]
#x_test=x_test.iloc[:,-2*n_comp:]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)


############################################### modeling  1 xgb #########################################

y_mean = np.mean(y_train)
'''
for c in [0.6]:
  print('*******************')
  print('colsample_bytree:',c)
  xgb_params = {
    #'n_trees': 500, 
    'booster':'gbtree',
    'eta': 0.005,
    'gamma':0.001,
    'min_child_weight':1,
    'max_depth': 3,
    'subsample': 0.97,
    'colsample_bytree':0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
  }

  # form DMatrices for Xgboost training
  dtrain = xgb.DMatrix(x_train, y_train)
  dtest = xgb.DMatrix(x_test)

  # FIND the best  num_boost_round for xgb
  # xgboost, cross-validation
  cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=1500, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   #verbose_eval=50, 
                   show_stdv=True
                  )

  num_boost_rounds = len(cv_result)
  print(num_boost_rounds)
  print('******************************')
'''

'''
# train single model--------------------------------------------------------------
# prepare dict of params for xgboost to run with

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)


xgb_params = {
    #'n_trees': 500, 
    'booster':'gbtree',
    'eta': 0.005,
    'gamma':0.005,
    'max_depth': 4,
    'subsample': 0.9,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1}

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=900)
# make predictions and save results
y_pred = model.predict(dtest)
output = pd.DataFrame({'id': test0['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('E:\\xgboost-depth{}-pca-ica.csv'.format(xgb_params['max_depth']), index=False)
'''



####################################################### modeling 2 DNN ########################################
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout

'''
#样本的随机抽样作为测试样本
import random
index=list(range(len(x_train)))
test_index = random.sample(index, 1000)##test_index is the index of test data随机选出2000个样本作为测试样本
train_index=[]
for i in range(len(x_train)):
  if i not in test_index:
    train_index.append(i)

x1=x_train.loc[train_index].as_matrix()
y1=y_train.loc[train_index].as_matrix()
print(y1)
c=1/0
x2=x_train.loc[test_index].as_matrix()
y2=y_train.loc[test_index]
'''

model = Sequential()
model.add(Dense(300, activation='relu',input_dim=x_train.shape[1]))
model.add(Dense(100, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='relu'))


model.compile(loss = 'mse', optimizer = 'adam',metrics=['mae'])
model.fit(x_train.as_matrix(), y_train.as_matrix(), nb_epoch = 10, batch_size = 30 )

model.save_weights('e:\\benzs\\my_model_weights.h5')
model.load_weights('e:\\benzs\\my_model_weights.h5') 
  
y_pred = model.predict(x_test.as_matrix()).reshape(len(x_test)) 
output = pd.DataFrame({'id': test0['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('E:\\dnn.csv', index=False)
'''
print('score of model:.........')

pred=model.predict(x2).reshape(len(x2)) 

k=0

p=[]
t=[]

for i in range(len(pred)):
  k=k+abs(pred[i]-y2[i])
  print(abs(pred[i]-y2[i]))
  t.append(y2[i])
  p.append(pred[i])
    
pd.Series(t).plot()
pd.Series(p).plot()
plt.show()

print(1-k/len(test_index))

'''





############################################ ensemble modeling #############################
def bagging_model(n_model,sample_factor,x_train,ly_train,x_test):
  import random
  y_mean = np.mean(y_train)
  n=[500,550,600,650,700,750,800,850,900,950]

  xgb_params = {
    #'n_trees': 500, 
    'booster':'gbtree',
    'eta': 0.005,
    'gamma':0.005,
    'min_child_weight':1,
    'max_depth': 4,
    'subsample': 0.9,
    'colsample_bytree':1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1}

  dtrain = xgb.DMatrix(x_train, y_train)
  dtest = xgb.DMatrix(x_test)

  model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=900)
  y_pred = model.predict(dtest)/(n_model+1)

  for i in range(n_model):

    sample_index = random.sample(list(range(len(x_train))) , int(sample_factor*len(x_train)))
    
    dtrain = xgb.DMatrix(x_train.loc[sample_index], y_train.loc[sample_index])
    dtest = xgb.DMatrix(x_test)

    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=n[i])
    y_pred = y_pred + model.predict(dtest)/(n_model+1)
    print(i)

  return y_pred



'''
pred = bagging_model(10,0.9,x_train,y_train,x_test)
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': pred})
output.to_csv('E:\\xgboost-bagging.csv', index=False)
'''

'''
重要参数： scale_pos_weight = nagtive_samples/positive_samples
含义：二分类中正负样本比例失衡，需要设置正样本的权重。比如当正负样本比例为1:10时，可以调节scale_pos_weight=10。
'''


