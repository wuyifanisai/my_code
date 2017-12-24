import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

'''
建立一个分类器，用以对样本进行分类
'''

df=pd.read_csv('E:\\Breast Cancer Wisconsin (Diagnostic) Data Set\\data_1.csv',header=0)
label=pd.read_csv('E:\\Breast Cancer Wisconsin (Diagnostic) Data Set\\label.csv',header=None)

###################################################### feature selecting ####################################
top_fea=5
model = RandomForestClassifier(random_state=1)
model.fit(df, label)
feature_imp = pd.DataFrame(model.feature_importances_, index=df.columns, columns=["importance"])
feat_rf = feature_imp.sort_values("importance", ascending=False).head(top_fea).index


names=df.columns
# sort importances
indices = np.argsort(model.feature_importances_)
# plot as bar chart
plt.barh(np.arange(len(names)), model.feature_importances_[indices])
plt.yticks(np.arange(len(names)) + 0.25, np.array(names)[indices])
_ = plt.xlabel('Relative importance')
plt.show()
#reduce features
features = np.hstack([feat_rf])
features = list(np.unique(features))
df=df[features]

############################ prepare the data ########################################

df=df.as_matrix()
#label=label.as_matrix()
label=np.array(label).reshape(1,569)[0]


df_train,df_test,label_train,label_test = df[:400],df[400:],label[:400],label[400:]

#rf------------------------------------------
rf=RandomForestClassifier(n_estimators=50,criterion='gini',max_depth=5,class_weight={0:1,1:1},max_features=0.8)

rf.fit(df_train,label_train)
label_pred = rf.predict(df_test)
print(accuracy_score(label_pred,label_test))

# xgb -------------------------------------
import xgboost as xgb
xgb_params = {
    'eta': 0.004,
    'gamma':0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'binary:logistic',  #这个参数代表是目标函数，根据(多)分类或回归不同的问题，选用不同的目标函数
    'eval_metric': 'error',      #误差评估函数，回归可以用rmse，分类可以用
   # 'verbose_eval':10,
   # 'base_score': y_mean,
    'silent': 1
}
dtrain = xgb.DMatrix(df_train, label_train)
dtest = xgb.DMatrix(df_test,label_test)

watchlist = [(dtest, 'eval'), (dtrain, 'train')]

model = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist)
y_pred = model.predict(dtest)

print(y_pred)
from sklearn.metrics import accuracy_score
print(accuracy_score(pd.Series(y_pred).apply(lambda x: x>0.5),label_test))



####################################### 模型评估  ###################################

def bin_cv(model, X_train, y_train):
	score= np.sqrt(cross_val_score(model, X_train, y_train, cv=3))
	return(score)

# LR ------------------------------------------
cs=[0.005,0.01,0.1,1.0,5,10,100]

result=[bin_cv(LR(penalty='l2',tol=0.0001,C=c,solver='liblinear',max_iter=500, ), df,label).mean() for c in cs]
print(result)
pd.Series(result).plot()
plt.title('LR with different C')
xlabels=cs
plt.xticks(range(1, 5), xlabels, rotation = 0) #坐标标签
plt.show()
result.sort()
print(result[-1])

model=LR(penalty='l2',tol=0.0001,C=.005,solver='liblinear',max_iter=500, )
model.fit(df,label)
print(features,model.coef_)

model2=RLR(C=1,scaling=0.5,sample_fraction=0.6,selection_threshold=0.3,n_resampling=100)
model2.fit(df,label)
print(model2.get_support())


# dt ------------------------------------------
d=[1,2,3,4,5,10,15,20,25]
m=['gini','entropy']

result=[bin_cv(DTC(criterion='entropy',max_depth=depth),df,label).mean() for depth in d]
print(result)
pd.Series(result).plot()
plt.title('dt with different depth')
xlabels=d
plt.xticks(range(1, len(d)), xlabels, rotation = 0) #坐标标签
plt.show()
result.sort()
print(result[-1])

model=DTC(criterion='entropy',max_depth=4)
model.fit(df,label)
C=1/0
from IPython.display import Image
from sklearn import tree

with open('E:\\Breast Cancer Wisconsin (Diagnostic) Data Set\\tree.dot','w') as f:
	f=tree.export_graphviz(model,out_file=f)

import os
os.unlink('E:\\Breast Cancer Wisconsin (Diagnostic) Data Set\\tree.dot')

import pydotplus
dot_data=tree.export_graphviz(model,out_file=None)
graph=pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('E:\\Breast Cancer Wisconsin (Diagnostic) Data Set\\tree.pdf')





dot_data=tree.export_graphviz(model,
							out_file=None,
							feature_names=['area_worst', 'concave points_mean', 'concavity_mean', 'perimeter_worst', 'radius_worst'],
							class_names=['0','1'],
							filled=True,
							rounded = True,
							special_characters=True)
graph=pyotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# DNN ---------------------------------------
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout

index0=[]
index1=[]

for i in range(len(label)):
	if label[i] ==0:
		index0.append(i)
	else:
		index1.append(i)


import random
index=list(range(len(df)))
train_index = random.sample(index0,int(0.8*len(index0))) + random.sample(index1,int(0.8*len(index1)))  ##test_index is the index of test data随机选出2000个样本作为测试样本
test_index=[]   ##train_index is the index of train data
for i in index:
	if i not in train_index:
		test_index.append(i)
print(len(train_index))


model = Sequential()
model.add(Dense(output_dim=50, input_dim=len(df[0]), activation='relu'))
model.add(Dense(output_dim=20, input_dim=50, activation='relu'))
model.add(Dense(output_dim=1, input_dim=20, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
model.fit(df[train_index], label[train_index], nb_epoch = 1000, batch_size = 20) 

pred = model.predict_classes(df[test_index]).reshape(len(test_index))
print(pred)

k=0
for i in range(len(pred)):
	if pred[i]==label[test_index][i]:
		k=k+1 
print(k/len(test_index))

#model.save_weights('E:\...\my_model_weights.h5')

'''
神经网络的精度还是比较高的，并且网络的规模不需要太大，太大反而容易过拟合降低精度

特征的选择会带来精度上的微小损失

好像特征减少对神经网络的影响较大
'''



