import tensorflow as tf 
c=1/0



import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, chi2
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, AdaBoostClassifier
'''
data=pd.read_csv('E:\\kaggle_KOBE\\data.csv')
data=data.iloc[:,:]#从原始数据中选取前1000条先进行分析
print(data.iloc[0,:])

#统计action_type的种类############################
action_type=[]
for i in range(len(data)):
	if data['action_type'][i] not in action_type:
		action_type.append(data['action_type'][i])
print(action_type)
print('kobe的得分方式种类数量：',len(action_type))

#统计COMBINE_action_type的种类############################
combine_action_type=[]
for i in range(len(data)):
	if data['combined_shot_type'][i] not in combine_action_type:
		combine_action_type.append(data['combined_shot_type'][i])
print(combine_action_type)
print('kobe的combine得分方式种类数量：',len(combine_action_type))

###########统计科比在每一种得分方式下的成功率#######################
shot_try_count={}#存放各种得分方式的出手次数
shot_made_count={}#存放各种得分方式的出手并成功次数
shot_type_success={}#存放各种得分方式的出手成功概率
shot_type_label={}#存放各种得分方式标记
#初始化
for type in action_type:
	shot_try_count[type]=0

for type in action_type:
	shot_made_count[type]=0

for i in range(len(data)):
	if data['shot_made_flag'][i]>=0:
		shot_try_count[data['action_type'][i]]=shot_try_count[data['action_type'][i]]+1
		if data['shot_made_flag'][i]==1:
			shot_made_count[data['action_type'][i]]=shot_made_count[data['action_type'][i]]+1

for type in action_type:
	print(type)
	if shot_try_count[type]>0:
		shot_type_success[type]=shot_made_count[type]/shot_try_count[type]
	else:
		shot_type_success[type]=0 #训练集合中没有出现过的得分动作的成功率设为0

print(shot_type_success)

for type in action_type:
	if shot_type_success[type]>0.8:
		shot_type_label[type]=7
	elif shot_type_success[type]>0.7:
		shot_type_label[type]=6
	elif shot_type_success[type]>0.6:
		shot_type_label[type]=5
	elif shot_type_success[type]>0.5:
		shot_type_label[type]=4
	elif shot_type_success[type]>0.4:
		shot_type_label[type]=3
	elif shot_type_success[type]>0.3:
		shot_type_label[type]=2
	else:
		shot_type_label[type]=0
print(shot_type_label)

#将data中的action——type变换过来
for i in range(len(data)):
	data['action_type'][i]=shot_type_label[data['action_type'][i]]
	print(i)
print(data['action_type'])
data.to_excel('E:\\kaggle_KOBE\\data01.xls')#保存将action_type特征转换为0-7数值的数据
'''

'''
#####################对出手位置的相关数据进行预处理##########################
data=pd.read_excel('E:\\kaggle_KOBE\\data01.xls')

#loc—x
data['loc_x']=(data['loc_x']-data['loc_x'].mean())/data['loc_x'].std()
print(data['loc_x'])

#loc_y
data['loc_y']=(data['loc_y']-data['loc_y'].mean())/data['loc_y'].std()
print(data['loc_y'])

#shot_distance
print(data['shot_distance'].describe())
data['shot_distance']=(data['shot_distance']-data['shot_distance'].mean())/data['shot_distance'].std()
print(data['shot_distance'])

#shot_zone_area
print(data['shot_zone_area'])
shot_zone_area=[]
for i in range(len(data)):
	if data['shot_zone_area'][i]=='Left Side(L)':
		data['shot_zone_area'][i]=0
	elif data['shot_zone_area'][i]=='Left Side Center(LC)':
		data['shot_zone_area'][i]=1
	elif data['shot_zone_area'][i]=='Right Side Center(RC)':
		data['shot_zone_area'][i]=2
	elif data['shot_zone_area'][i]=='Center(C)':
		data['shot_zone_area'][i]=3
	elif data['shot_zone_area'][i]=='Right Side(R)':
		data['shot_zone_area'][i]=4
	else:
		data['shot_zone_area'][i]=5 #'Right Side(LR)'
	print(i)
print(data['shot_zone_area'])


data.to_excel('E:\\kaggle_KOBE\\data02.xls')#保存转换过loc_x,loc_y,shot_distance,shot_zone_area的数据
'''

'''
######################对时间相关的特征进行预处理####################################
data=pd.read_excel('E:\\kaggle_KOBE\\data02.xls')
print(data.columns)
print(data['game_date'][0])

#gamedata特征的处理
for i in range(len(data)):
	data['game_date'][i]=(int(data['game_date'][i][:4])-1996)*12+int(data['game_date'][i][5:7])-11+1
	print(data['game_date'][i])
	#1996.11---1997.5==>(1997-1996)*12+(5)-11+1


#每一节比赛剩余分钟数minutes_remaining特征的处理

#比赛节数特征的处理period

##################2分球，3分球特征shot_type的预处理#################################
print(data['shot_type'])
for i in range(len(data)):
	data['shot_type'][i]=int(data['shot_type'][i][0])
	print(i)
data.to_excel('E:\\kaggle_KOBE\\data03.xls')
'''

'''
#######################处理combined shot type#######################################
data=pd.read_excel('E:\\kaggle_KOBE\\data03.xls')
print(data.iloc[0,:])

t=[]
for i in range(len(data)):
	if data['combined_shot_type'][i] not in t:
		t.append(data['combined_shot_type'][i])
print(t)
for i in range(len(data)):
	data['combined_shot_type'][i]=t.index(data['combined_shot_type'][i])
	print(i)
#######################处理shot_zone_basic#######################################
b=[]
for i in range(len(data)):
	if data['shot_zone_basic'][i]=='Restricted Area':
		data['shot_zone_basic'][i]=1
	elif data['shot_zone_basic'][i]=='Mid-Range':
		data['shot_zone_basic'][i]=2
	elif data['shot_zone_basic'][i]== 'In The Paint (Non-RA)':
		data['shot_zone_basic'][i]=3
	elif data['shot_zone_basic'][i]== 'Above the Break 3':
		data['shot_zone_basic'][i]=4
	elif data['shot_zone_basic'][i]== 'Right Corner 3':
		data['shot_zone_basic'][i]=5
	elif data['shot_zone_basic'][i]== 'Left Corner 3':
		data['shot_zone_basic'][i]=6
	else:
		data['shot_zone_basic'][i]=7

#######################处理shot_zone_range#######################################
r=[]
for i in range(len(data)):
	if data['shot_zone_range'][i]=='Less Than 8 ft.':
		data['shot_zone_range'][i]=1
	elif data['shot_zone_range'][i]=='8-16 ft.':
		data['shot_zone_range'][i]=2
	elif data['shot_zone_range'][i]=='16-24 ft.':
		data['shot_zone_range'][i]=3
	elif data['shot_zone_range'][i]=='24+ ft.':
		data['shot_zone_range'][i]=4
	else:
		data['shot_zone_range'][i]=5

data.to_excel('E:\\kaggle_KOBE\\data04.xls')
'''
'''
############opponent特征处理###############################
data=pd.read_excel('E:\\kaggle_KOBE\\data04.xls')

t=[]
for i in range(len(data)):
	if data['opponent'][i] not in t:
		t.append(data['opponent'][i])
print(t)
for i in range(len(data)):
	data['opponent'][i]=t.index(data['opponent'][i])+1
	print(i)
data.to_excel('E:\\kaggle_KOBE\\data05.xls')
'''
'''
################lat特征处理##################################
data=pd.read_excel('E:\\kaggle_KOBE\\data05.xls')

data[['lat','loc_x','loc_y','lon','shot_distance','minutes_remaining','seconds_remaining']]=(data[['lat','loc_x','loc_y','lon','shot_distance','minutes_remaining','seconds_remaining']]-data[['lat','loc_x','loc_y','lon','shot_distance','minutes_remaining','seconds_remaining']].min())/(data[['lat','loc_x','loc_y','lon','shot_distance','minutes_remaining','seconds_remaining']].max()-data[['lat','loc_x','loc_y','lon','shot_distance','minutes_remaining','seconds_remaining']].min())
print(data.iloc[0,:])
data.to_excel('E:\\kaggle_KOBE\\data06.xls')
'''
'''
###################game date 特征处理#################################
data=pd.read_excel('E:\\kaggle_KOBE\\data06.xls')
data['month']=data['shot_id']
data['year']=data['shot_id']
for i in range(len(data)):
	s=str(data['game_date'][i])
	data['year'][i]=int(s[:4])-1995
	data['month'][i]=int(s[5:7])
	print(i)

data.to_excel('E:\\kaggle_KOBE\\data07.xls')
'''
'''
#########################lat 离散化##############################
data=pd.read_excel('E:\\kaggle_KOBE\\data07.xls')
k = 20
d = pd.cut(data['lat'], k, labels = range(k)) #等宽离散化，各个类比依次命名为0,1,2,3
data['lat']=d
#########################loc_x,y 离散化##############################
k = 20
d = pd.cut(data['loc_x'], k, labels = range(k)) #等宽离散化，各个类比依次命名为0,1,2,3
data['loc_x']=d

k = 20
d = pd.cut(data['loc_y'], k, labels = range(k)) #等宽离散化，各个类比依次命名为0,1,2,3
data['loc_y']=d

#########################lon minutes,seconds,shot distance 离散化##############################
k = 20
d = pd.cut(data['lon'], k, labels = range(k)) #等宽离散化，各个类比依次命名为0,1,2,3
data['lon']=d

k = 20
d = pd.cut(data['minutes_remaining'], k, labels = range(k)) #等宽离散化，各个类比依次命名为0,1,2,3
data['minutes_remaining']=d

k = 20
d = pd.cut(data['seconds_remaining'], k, labels = range(k)) #等宽离散化，各个类比依次命名为0,1,2,3
data['seconds_remaining']=d

k = 20
d = pd.cut(data['shot_distance'], k, labels = range(k)) #等宽离散化，各个类比依次命名为0,1,2,3
data['shot_distance']=d
print(data.iloc[0,:])

data.to_excel('E:\\kaggle_KOBE\\data08_init.xls')
'''

'''
########################可视化数据#############################################
data=pd.read_excel('E:\\kaggle_KOBE\\data08.xls')

#投篮中与不中的分布
ax = plt.axes()
sns.countplot(x='shot_made_flag', data=data, ax=ax);
ax.set_title('Target class distribution')
plt.show()

#集中特征的散点图
sns.pairplot(data, vars=['loc_x', 'loc_y', 'lat', 'lon', 'shot_distance'], hue='shot_made_flag', size=3)
plt.show()


#各个特征对于shotmadeflag的箱形图
f, axarr = plt.subplots(4, 2, figsize=(15, 15))
sns.boxplot(x='lat', y='shot_made_flag', data=data, showmeans=True, ax=axarr[0,0])
sns.boxplot(x='lon', y='shot_made_flag', data=data, showmeans=True, ax=axarr[0, 1])
sns.boxplot(x='loc_y', y='shot_made_flag', data=data, showmeans=True, ax=axarr[1, 0])
sns.boxplot(x='loc_x', y='shot_made_flag', data=data, showmeans=True, ax=axarr[1, 1])
sns.boxplot(x='minutes_remaining', y='shot_made_flag', showmeans=True, data=data, ax=axarr[2, 0])
sns.boxplot(x='seconds_remaining', y='shot_made_flag', showmeans=True, data=data, ax=axarr[2, 1])
sns.boxplot(x='shot_distance', y='shot_made_flag', data=data, showmeans=True, ax=axarr[3, 0])

axarr[0, 0].set_title('Latitude')
axarr[0, 1].set_title('Longitude')
axarr[1, 0].set_title('Loc y')
axarr[1, 1].set_title('Loc x')
axarr[2, 0].set_title('Minutes remaining')
axarr[2, 1].set_title('Seconds remaining')
axarr[3, 0].set_title('Shot distance')

plt.tight_layout()
plt.show()

#各个特征对于投篮是否命中的分布图
f, axarr = plt.subplots(8, figsize=(15, 25))

sns.countplot(x="combined_shot_type", hue="shot_made_flag", data=data, ax=axarr[0])

sns.countplot(x="period", hue="shot_made_flag", data=data, ax=axarr[1])
sns.countplot(x="playoffs", hue="shot_made_flag", data=data, ax=axarr[2])
sns.countplot(x="shot_type", hue="shot_made_flag", data=data, ax=axarr[3])
sns.countplot(x="shot_zone_area", hue="shot_made_flag", data=data, ax=axarr[4])
sns.countplot(x="shot_zone_basic", hue="shot_made_flag", data=data, ax=axarr[5])
sns.countplot(x="shot_zone_range", hue="shot_made_flag", data=data, ax=axarr[6])

axarr[0].set_title('Combined shot type')
axarr[1].set_title('Period')
axarr[2].set_title('Playoffs')
axarr[3].set_title('Shot Type')
axarr[4].set_title('Shot Zone Area')
axarr[5].set_title('Shot Zone Basic')
axarr[6].set_title('Shot Zone Range')

plt.tight_layout()
plt.show()
'''
'''
#####################create new feature #########################################
data=pd.read_excel('E:\\kaggle_KOBE\\data08.xls')
#create a new feature :last_5_sec_in_period
data['seconds_from_period_end'] = 60 * data['minutes_remaining'] + data['seconds_remaining']
data['last_5_sec_in_period'] = (data['seconds_from_period_end'] < 5).astype('int')
data.drop('seconds_from_period_end', axis=1, inplace=True)
print(data['last_5_sec_in_period'])
data.to_excel('E:\\kaggle_KOBE\\data08.xls')
'''

'''
#create a new feature : home_play
data['home_play'] = data['matchup'].str.contains('vs').astype('int')
data.drop('matchup', axis=1, inplace=True)


print(data.iloc[0,:])
data.to_excel('E:\\kaggle_KOBE\\data08.xls')
'''

'''
######################Encode categorical variables###############################
data=pd.read_excel('E:\\kaggle_KOBE\\data08.xls')
print(data.columns)
categorial_cols = [
    'action_type', 'combined_shot_type', 'lat', 'loc_x', 'loc_y', 'lon',
        'period', 'playoffs', 'shot_distance',
        'shot_type', 'shot_zone_area', 'shot_zone_basic',
       'shot_zone_range', 'opponent',  'month', 'year', 'home_play',
       'last_5_sec_in_period']


for cc in categorial_cols:
    dummies = pd.get_dummies(data[cc])
    dummies = dummies.add_prefix("{}#".format(cc))
    data.drop(cc, axis=1, inplace=True)
    data = data.join(dummies)


print(data.iloc[0,:])
data.to_excel('E:\\kaggle_KOBE\\data09_feature01.xls')
'''
'''
###############feature selection：reduce the number of the features #########################################
data=pd.read_excel('E:\\kaggle_KOBE\\data09_feature01.xls')
#提取出flag为空的记录
unknown_mask = data['shot_made_flag'].isnull()

#提取出训练输入X，训练目标Y，测试输入DATASUBMIT
datasubimt=data[unknown_mask]
datasubimt.drop('shot_made_flag', axis=1, inplace=True)
datasubimt.drop('shot_id', axis=1, inplace=True)

data_train=data[~unknown_mask]
Y=data_train['shot_made_flag']
data_train.drop('shot_made_flag', axis=1, inplace=True)
data_train.drop('shot_id', axis=1, inplace=True)
X=data_train
'''
'''
#Variance Threshold SELECT FEATURE##############################

threshold = 0.90
vt = VarianceThreshold().fit(X)

# Find feature names
feat_var_threshold = data.columns[vt.variances_ > threshold * (1-threshold)]
print(feat_var_threshold)


#Top 20 most important features using RF model ######################################

model = RandomForestClassifier()
model.fit(X, Y)

feature_imp = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["importance"])
feat_imp_20 = feature_imp.sort_values("importance", ascending=False).head(20).index
print(feat_imp_20)


# Univariate feature selection ########################################################

X_minmax = MinMaxScaler(feature_range=(0,1)).fit_transform(X)
X_scored = SelectKBest(score_func=chi2, k='all').fit(X_minmax, Y)
feature_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': X_scored.scores_
    })

feat_scored_20 = feature_scoring.sort_values('score', ascending=False).head(20)['feature'].values
print(feat_scored_20)

 
# Recursive Feature Elimination #######################################################

rfe = RFE(LogisticRegression(), 20)
rfe.fit(X, Y)

feature_rfe_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': rfe.ranking_
    })

feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values
print(feat_rfe_20)


###############################################
features = np.hstack([
        feat_var_threshold, 
        feat_imp_20,
        feat_scored_20,
        feat_rfe_20
    ])

features = np.unique(features)
print(features)
'''

'''
features=['action_type#0', 'action_type#2' ,'action_type#3' ,'action_type#4',
 'action_type#6', 'action_type#7', 'combined_shot_type#0',
 'combined_shot_type#1' ,'combined_shot_type#2', 'home_play#0', 'home_play#1',
 'last_5_sec_in_period#1' ,'lat#12' ,'lat#13', 'lat#14', 'lat#15' ,'lat#16',
 'lat#18', 'lat#8', 'lat#9' ,'loc_x#10', 'loc_x#19' ,'loc_x#8' ,'loc_y#0',
 'loc_y#1' ,'loc_y#10' ,'loc_y#11' ,'loc_y#2' ,'loc_y#3' ,'lon#10' ,'lon#18',
 'lon#19' ,'lon#8' ,'minutes_remaining' ,'month#1' ,'month#10' ,'month#11',
 'month#12' ,'month#2' ,'month#3', 'month#4' ,'month#6' ,'opponent#32',
 'opponent#33', 'period#1' ,'period#2', 'period#3' ,'period#4' ,'period#6',
 'period#7', 'playoffs#0', 'shot_distance#0' ,'shot_distance#1',
 'shot_distance#10', 'shot_distance#11' ,'shot_distance#18',
 'shot_distance#19' ,'shot_distance#2', 'shot_distance#3', 'shot_distance#4',
 'shot_distance#6' ,'shot_distance#8' ,'shot_distance#9',
 'shot_type#2', 'shot_type#3' ,'shot_zone_area#0', 'shot_zone_area#1',
 'shot_zone_area#2', 'shot_zone_area#3' ,'shot_zone_area#4',
 'shot_zone_area#5' ,'shot_zone_basic#1' ,'shot_zone_basic#2',
 'shot_zone_basic#4' ,'shot_zone_basic#6', 'shot_zone_basic#7',
 'shot_zone_range#1' ,'shot_zone_range#2', 'shot_zone_range#4',
 'shot_zone_range#5' ,'year#12','year#17' ,'year#20', 'year#21' ,'year#7']

X= X[features]
datasubimt=datasubimt[features]
print(datasubimt)

X.to_excel('E:\\kaggle_KOBE\\data09_X_feature01.xls')
pd.DataFrame(Y).to_excel('E:\\kaggle_KOBE\\data09_Y_feature01.xls')
datasubimt.to_excel('E:\\kaggle_KOBE\\data09_datasubimt_feature01.xls')
'''

'''
######### feature PCA ##################################
X=pd.read_excel('E:\\kaggle_KOBE\\data09_X_feature01.xls')
Y=pd.read_excel('E:\\kaggle_KOBE\\data09_Y_feature01.xls')
datasubimt=pd.read_excel('E:\\kaggle_KOBE\\data09_datasubimt_feature01.xls')

components = 10
pca = PCA(n_components=components).fit(X)

pca_variance_explained_df = pd.DataFrame({
    "component": np.arange(1, components+1),
    "variance_explained": pca.explained_variance_ratio_            
    })

ax = sns.barplot(x='component', y='variance_explained', data=pca_variance_explained_df)
ax.set_title("PCA - Variance explained")
plt.show()

X=pca.transform(X)
print(X)
'''


#############################################################################
#################  Evaluate Algorithms  ######################################

################################ DATA FROM DATA08 #############################################
data=pd.read_excel('E:\\kaggle_KOBE\\data08_init.xls')
print(data.columns)

data_train=data[data['shot_made_flag'].notnull()]
data_test=data[data['shot_made_flag'].isnull()]

x_train=data_train[['action_type', 'combined_shot_type', 'lat', 'loc_x', 'loc_y', 'lon',
       'minutes_remaining', 'period', 'playoffs', 'seconds_remaining',
       'shot_distance',  'shot_type', 'shot_zone_area',
       'shot_zone_basic', 'shot_zone_range', 'opponent', 'month',
       'year']]
y_train=data_train[['shot_made_flag']]

x_test=data_test[['action_type', 'combined_shot_type', 'lat', 'loc_x', 'loc_y', 'lon',
       'minutes_remaining', 'period', 'playoffs', 'seconds_remaining',
       'shot_distance',  'shot_type', 'shot_zone_area',
       'shot_zone_basic', 'shot_zone_range', 'opponent', 'month',
       'year']]

################################## DATA FROM DATA09  ################################################
X=pd.read_excel('E:\\kaggle_KOBE\\data09_X_feature01.xls')
Y=pd.read_excel('E:\\kaggle_KOBE\\data09_Y_feature01.xls')
datasubmit=pd.read_excel('E:\\kaggle_KOBE\\data09_datasubimt_feature01.xls')

#FOR CV--------
seed = 7
processors=1
num_folds=5
num_instances=len(X)
scoring='log_loss'
kfold = KFold(n=num_instances, n_folds=num_folds, random_state=seed)
#-------------------
'''
########## BAGGED DTC CV FOR Evaluate Algorithms ##############
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
print('BAGGED DTC CV FOR Evaluate Algorithms:',"({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))

########## Random Forest CV FOR Evaluate Algorithms ##############
num_trees = 100
num_features = 10
model = RandomForestClassifier(n_estimators=num_trees, max_features=num_features)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
print('RF CV FOR Evaluate Algorithms:',"({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))

########## Extra Trees CV FOR Evaluate Algorithms ##############
num_trees = 100
num_features = 10
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=num_features)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
print('Extra Trees CV FOR Evaluate Algorithms:',"({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))

########## LR CV FOR Evaluate Algorithms ##############
penalty='l2'
C=10.0
max_iter=100
model = LogisticRegression(penalty='l2',C=10.0,max_iter=100)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
print('LR CV FOR Evaluate Algorithms:',"({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))

########## ADABOOST CV FOR Evaluate Algorithms ##############
model = AdaBoostClassifier(n_estimators=100, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
print('ADABOOST CV FOR Evaluate Algorithms:',"({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))

########## GBDT CV FOR Evaluate Algorithms ##############
model = GradientBoostingClassifier(n_estimators=100, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
print('GBDT CV FOR Evaluate Algorithms:',"({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))
'''

#################################################################################################
##############################  Hyperparameter tuning  ###########################################
'''
BAGGED DTC CV FOR Evaluate Algorithms: (-0.757) +/- (0.014)
RF CV FOR Evaluate Algorithms: (-0.768) +/- (0.010)
Extra Trees CV FOR Evaluate Algorithms: (-2.849) +/- (0.451)
LR CV FOR Evaluate Algorithms: (-0.610) +/- (0.005)
ADABOOST CV FOR Evaluate Algorithms: (-0.691) +/- (0.000)
GBDT CV FOR Evaluate Algorithms: (-0.607) +/- (0.005)
'''

'''
##Logistic Regression GridSearchCV --------------------------------------
lr_grid = GridSearchCV(
    estimator = LogisticRegression(random_state=seed),
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 1, 10, 100, 1000]
    }, 
    cv = kfold, 
    scoring = scoring, 
    n_jobs = processors)
lr_grid.fit(X, Y)
print(lr_grid.best_score_)
print(lr_grid.best_params_)
'''
# BEST: {'penalty': 'l2', 'C': 1}

'''
## RF GridSearchCV --------------------------------------------------------
rf_grid = GridSearchCV(
    estimator = RandomForestClassifier(warm_start=True, random_state=seed),
    param_grid = {
        'n_estimators': [100, 150, 200],
        'criterion': ['gini', 'entropy'],
        'max_features': [0.5,0.7,1.0],
        'max_depth': [8,9,10],
        'bootstrap': [True]
    }, 
    cv = kfold, 
    scoring = scoring, 
    n_jobs = processors)
rf_grid.fit(X, Y)
print(rf_grid.best_score_)
print(rf_grid.best_params_)
'''
# best:{'n_estimators': 200, 'max_features': 0.5, 'bootstrap': True, 'criterion': 'entropy', 'max_depth': 8}

'''
## adaboost GridSearchCV --------------------------------------------------------
ada_grid = GridSearchCV(
    estimator = AdaBoostClassifier(random_state=seed),
    param_grid = {
        'algorithm': ['SAMME', 'SAMME.R'],
        'n_estimators': [10, 25, 50],
        'learning_rate': [1e-3, 1e-2, 1e-1,0.5]
    }, 
    cv = kfold, 
    scoring = scoring, 
    n_jobs = processors)
ada_grid.fit(X, Y)
print(ada_grid.best_score_)
print(ada_grid.best_params_)
#best:{'learning_rate': 0.001, 'algorithm': 'SAMME.R', 'n_estimators': 25}
'''
'''
## GBDT GridSearchCV --------------------------------------------------------
gbm_grid = GridSearchCV(
    estimator = GradientBoostingClassifier(warm_start=True, random_state=seed),
    param_grid = {
        'n_estimators': [100, 150,200],
        'max_depth': [8,9,10],
        'max_features': [0.5,0.7,1.0],
        'learning_rate': [1e-1,0.5,1]
    }, 
    cv = kfold, 
    scoring = scoring, 
    n_jobs = processors)
gbm_grid.fit(X, Y)
print(gbm_grid.best_score_)
print(gbm_grid.best_params_)

#best:{'learning_rate': 0.1, 'max_depth': 8, 'max_features': 0.5, 'n_estimators': 100}
'''

'''
## GBDT GridSearchCV --------------------------------------------------------
gbm_grid = GridSearchCV(
    estimator = XGBClassifier(),
    param_grid = {
        'n_estimators': [100, 150,200],
        'max_depth': [6,9],
        'learning_rate': [1e-1,0.5,1]
    }, 
    cv = kfold, 
    scoring = scoring, 
    n_jobs = processors)
gbm_grid.fit(X, Y)
print(gbm_grid.best_score_)
print(gbm_grid.best_params_)

'''

########################### Voting ensemble ##################################
# Create sub models
estimators = []
estimators.append(('lr', LogisticRegression(penalty='l2', C=1)))
estimators.append(('gbm', GradientBoostingClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, max_features=0.5, warm_start=True, random_state=seed)))
estimators.append(('rf', RandomForestClassifier(bootstrap=True, max_depth=8, n_estimators=200, max_features=0.5, criterion='entropy', random_state=seed)))
estimators.append(('ada', AdaBoostClassifier(algorithm='SAMME.R', learning_rate=1e-2, n_estimators=25, random_state=seed)))
estimators.append(('xgb', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)))

# create the ensemble model
ensemble = VotingClassifier(estimators, voting='soft', weights=[2,3,3,1,3])
# CV for Voting ensemble
results = cross_val_score(ensemble, x_train, y_train, cv=kfold, scoring=scoring,n_jobs=processors)
print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))


#########################################################################################################################################################################
##########################################################################################################################################################################



#####################################生成结果，用于提交########################
model=ensemble
model.fit(x_train,y_train)
#x=pd.DataFrame(x).as_matrix()#不进行DataFrame适用于xgboost
pred_proba_test=model.predict_proba(x_test)
sample=pd.read_csv('E:\\kaggle_KOBE\\sample_submission.csv')

for i in range(len(sample)):
	sample.iloc[i,1]=pred_proba_test[i][1]
sample.to_csv('E:\\kaggle_KOBE\\test_submission.csv')
