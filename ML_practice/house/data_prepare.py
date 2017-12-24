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


d_train=pd.read_csv('E:\\house\\train.csv')
d_test=pd.read_csv('E:\\house\\test.csv')
print('train_csv_shape:',d_train.shape)
print('test_csv_shape:',d_test.shape)
print('-----------------------------')
# check the head of date
print('the columns of d_train:')
print(d_train.columns)
print('the columns of d_test:')
print(d_test.columns)
print('------------------------------')

# concat the feature parts of train and test 
df=pd.concat((d_train.loc[:,'MSSubClass':'SaleCondition'],d_test.loc[:,'MSSubClass':'SaleCondition']))
print('df_shape:',df.shape)
print('-------------------------------')

#check the df_dtypes
print('value_counts of df_dtypes:')
print(df.dtypes)
print()
print(df.dtypes.value_counts())
print('------------------------------')

#get all kinds types of feture
int64_feats=list(df.dtypes[df.dtypes=='int64'].index)
print('int64_feats is:')
print(int64_feats)
print()
float64_feats=list(df.dtypes[df.dtypes=='float64'].index)
print('float64_feats is:')
print(float64_feats)
print()
object_feats=list(df.dtypes[df.dtypes=='object'].index)
print('object_feats is:')
print(object_feats)
print('-----------------------------')

#check the missing point in the data in every feature
nan_counts_train=d_train.isnull().sum(axis=0)
print('the number of nan of d_train:')
print(len(nan_counts_train.loc[nan_counts_train>0]))


nan_counts_test=d_test.isnull().sum(axis=0)
print('the number of nan of d_test:')
print(len(nan_counts_test.loc[nan_counts_test>0]))


nan_counts_df=df.isnull().sum(axis=0)
print('the number of nan of df:')
print(nan_counts_df.loc[nan_counts_df>0].shape[0])
print()
print('number of missing point of feature:')
print()
print(nan_counts_df.loc[nan_counts_df>0])
print('--------------------------------')


#find the num_type and obj_feat feture containing missing point
num_feat=int64_feats+float64_feats
nan_num_feats=[feat for feat in list(nan_counts_df.loc[nan_counts_df>0].index) if feat in num_feat]
print('find the num_type feture containing missing point:')
print(nan_num_feats)
print(len(nan_num_feats))
print()
nan_object_feats=[feat for feat in list(nan_counts_df.loc[nan_counts_df>0].index) if feat in object_feats]
print('find the object_type feture containing missing point:')
print(nan_object_feats)
print(len(nan_object_feats))
print('------------------------')


############################## deal with the NA ###########################################


na_obj_btw=[]#用来存放缺失值处理好的object特征

# ------->lotfrontage missing point------------------------------------------------------------------------------
#通过BldgType的取值对lotfrontage进行分组，对每组求出lotfrontage的平均值
mean_LotFrontage=df.groupby('BldgType').LotFrontage.mean() #是一个series
# 对相应的lotfrontage missing point 用其属于的BldgType分组中lotfrontage的平均值进行填充
for x in list(mean_LotFrontage.index):
	df.loc[(df.LotFrontage.isnull())&(df.BldgType==x),'LotFrontage']=mean_LotFrontage[x]
print('---------------------------')

#------->MasVnrArea missing point---------------------------------------------------------------------------------
#将'MasVnrArea','MasVnrType'缺失值一起处理
print(df.loc[df.MasVnrType.isnull(),['MasVnrArea','MasVnrType']])
print()
print(df.MasVnrType.value_counts(dropna=False))
#给那些很多的masvnrtype 的缺失值定义一个表示没有的NONE值
df.loc[df.MasVnrArea.isnull(),'MasVnrType']='None'
df.loc[df.MasVnrArea.isnull(),'MasVnrArea']=0

df.loc[df.MasVnrType.isnull(),'MasVnrType']='BrkFace'
print()
na_obj_btw.append('MasVnrType')
print('---------------------------')

#------>features related to basement地下室--------------------------------------------------------------------------
print('将features related to basement缺失值同时进行处理')
print()

basement_feats_num=['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
#输出上述每种特征对应的缺失值个数
print('输出上述basement_feats_num每种特征对应的缺失值个数:')
for x in basement_feats_num:
	print(x,'---->',df.loc[df[x].isnull(),:].shape[0])
print()
basement_feats_obj=['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
print('输出上述basement_feats_obj每种特征对应的缺失值个数:')
for x in basement_feats_obj:
	print(x,'-->',df.loc[df[x].isnull(),:].shape[0])
print()


print('查看每种basement相关的特征在缺失的行中，另外的basement相关的特征的情况.......')

#存在BsmtFullBath缺失值的行,basement_feats_num缺失值设为0
df.loc[df.BsmtFullBath.isnull(),basement_feats_num]=0

#存在BsmtCond缺失值的行,basement_feats_obj缺失值设为without
df.loc[df.BsmtCond.isnull(),basement_feats_obj]='without'

#存在Bsmtqual缺失值的行,Bsmtqual缺失值设为TA
df.loc[df.BsmtQual.isnull(),'BsmtQual']='TA'

#考察 BsmtUnfSF与 TotalBsmtSF，构造特征 bsmunfinshedratio 
df['bsmunfinshedratio']=df['BsmtUnfSF']/df['TotalBsmtSF']
df.groupby('BsmtExposure')['bsmunfinshedratio'].mean().sort_values()

#存在BsmtExposure缺失值的行,BsmtExposure缺失值设为NO
df.loc[df.BsmtExposure.isnull(),'BsmtExposure']='No'

#存在BsmtFinType2缺失值的行,BsmtFinType2缺失值设为NO
print(df.loc[df.BsmtFinType2.isnull(),basement_feats_obj+basement_feats_num])
print(df.groupby('BsmtFinType2')['BsmtFinSF2'].mean())
df.loc[df.BsmtFinType2.isnull(),'BsmtFinType2']='ALQ'
print()

#features related to basement 缺失值是否都补全
for x in basement_feats_obj+basement_feats_num:
	print(x,'--->',df.loc[df[x].isnull(),:].shape[0])

na_obj_btw=na_obj_btw+['BsmtQual','BsmtCond','BsmtFinType1']
print('---------------------------')

#------>features related to garage地下室-------------------------------------------------------------------------------
print('将features related to garage缺失值同时进行处理......')
print()
# again, it's better to deal with all garage features in the mean time
garage_feats_num = ['GarageYrBlt', 'GarageCars', 'GarageArea']
garage_feats_obj = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

print('输出上述garage_feats_num每种特征对应的缺失值个数:')
for x in garage_feats_num:
	print(x,'--->',df.loc[df[x].isnull(),:].shape[0])
print()
print('输出上述garage_feats_obj每种特征对应的缺失值个数:')
for x in garage_feats_obj:
	print(x,'--->',df.loc[df[x].isnull(),:].shape[0])
print()

#存在garagecars缺失值的行,garagecars and GarageArea 都设为0
#print(df.loc[df.GarageCars.isnull(),garage_feats_num+garage_feats_obj])
df.loc[df.GarageCars.isnull(),['GarageCars', 'GarageArea']]=0

#存在Garagetype缺失值的行，garage_feats_obj的缺失值设为 ’without‘
df.loc[df.GarageType.isnull(),garage_feats_obj]='without'
 

#'GarageFinish','GarageQual','GarageCond' 缺失的行进行补缺
df.loc[(df.GarageFinish.isnull())&(df.GarageCars==1), 'GarageFinish'] = 'Unf'
df.loc[(df.GarageQual.isnull())&(df.GarageCars==1), 'GarageQual'] = 'TA'
df.loc[(df.GarageCond.isnull())&(df.GarageCars==1), 'GarageCond'] = 'TA'
df.loc[(df.GarageFinish.isnull())&(df.GarageCars==0), 'GarageFinish'] = 'without'
df.loc[(df.GarageQual.isnull())&(df.GarageCars==0), 'GarageQual'] = 'without'
df.loc[(df.GarageCond.isnull())&(df.GarageCars==0), 'GarageCond'] = 'without'

#存在GarageYrBlt缺失的行中，GarageYrBlt设为0
df.loc[df.GarageYrBlt.isnull(), 'GarageYrBlt'] = 0


print('检查garcar相关特征值缺失值是否全部补齐。。')
for x in garage_feats_num+garage_feats_obj:
	print(x,'--->',df.loc[df[x].isnull(),:].shape[0])

na_obj_btw=na_obj_btw+garage_feats_obj
print('--------------------------------------------------------------')
print('查看所有的num_feat是否已经补缺完毕')
for x in nan_num_feats:
	print(x,'--->',df.loc[df[x].isnull(),:].shape[0])


#处理剩下的还存在缺失值的obj_feat的特征
nan_object_feats=[x for x in nan_object_feats if x not in na_obj_btw]
print('剩下的还存在缺失值的obj_feat的特征:')
print(nan_object_feats)
print()
for x in nan_object_feats:
	print(x,'--->',df.loc[df[x].isnull(),:].shape[0])
print('------------------------------------------------------------')

# 处理 MSZoning 缺失值
df.loc[df.MSZoning.isnull(),'MSZoning']='RL'

# 处理 MSZoning 缺失值
df.loc[df.Alley.isnull(),'Alley']='without'

# 处理 Utilities 缺失值
print(df.Utilities.mode())
df.loc[df.Utilities.isnull(),'Utilities']='AllPub'

#处理 Exterior1st ,Exterior2nd 缺失值
df.loc[df.Exterior1st.isnull(),['Exterior1st' ,'Exterior2nd']]='VinylSd'

#处理 Electrical 缺失值
df.loc[df.Electrical.isnull(), 'Electrical'] = 'SBrkr'


df.loc[df.KitchenQual.isnull(), 'KitchenQual'] = 'TA'


df.loc[df.Functional.isnull(), 'Functional'] = 'Typ'

df.loc[df.FireplaceQu.isnull(), 'FireplaceQu'] = 'without'

df.loc[df.PoolQC.isnull(), 'PoolQC'] = 'without'

df.loc[df.Fence.isnull(), 'Fence'] = 'without'

df.loc[df.MiscFeature.isnull(), 'MiscFeature'] = 'without'

df.loc[df.SaleType.isnull(), 'SaleType'] = 'WD'
print()

print('obj_feat 的缺失值已经全部补上')
for x in nan_object_feats:
	print(x,'-->',df.loc[df[x].isnull(),:].shape[0])
print('-------------------------------------------------------')
print('检查是否还有缺失值存在。。。。。')
df.drop(['bsmunfinshedratio'],inplace=True,axis=1)
nan_counts_df=df.isnull().sum(axis=0)
print(nan_counts_df[nan_counts_df>0])
print('--------------------------------------------------')




#######################################  进行特征工程  ##########################################

#以下是obj特征中经常出现的字符型特征---------------------------------------------------------------------------------------
ordinal_words = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
ordinal_feats=[] #里面存放包含2个或以上ordinal_word的obj类型的特征
for x in object_feats:
	flag=0
	for s in ordinal_words:
		if df.loc[df[x].str.contains(s,case=True),:].shape[0]>0:
			flag=flag+1
	if flag>=2:
		ordinal_feats.append(x)
print('包含2个或以上ordinal_word的obj类型的特征')
print(ordinal_feats)
print()

#每种ordinal_feat取值中ordinal_word的种类个数
for x in ordinal_feats:
	print(x,'--->',df[x].unique(),'--->',len(df[x].unique()))

#通过ordinal_dict将ordinal_feats的取值转换为数字型取值
ordinal_dict={'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'without':0}

ordinal_num_feats=[]
for x in ordinal_feats:
	df[x+'_num']=df[x].map(ordinal_dict)
	ordinal_num_feats.append(x+'_num')
print(ordinal_num_feats)
df.drop(ordinal_feats,axis=1,inplace=True)#########################


toDrop_feats=[]
#-------------------------------------------------------------------------------------------------------------------------
ordinal_obj_feats = ['PavedDrive', 'Functional', 'CentralAir', 'Fence', 'Utilities']

#将Functional的取值转换成数字，并依次构造新特征 Functional_num等等
df['Functional_num']=df.Functional.replace({'Typ': 6,'Min1': 5,'Min2': 5,'Mod': 4,'Maj1': 3,'Maj2': 3,'Sev': 2,'Sal': 1})
df['PavedDrive_num'] = df.PavedDrive.replace({'Y': 3, 'P': 2, 'N': 1})
df['CentralAir_num'] = df.CentralAir.replace({'Y': 1, 'N': 0})
df['Fence_num'] = df.CentralAir.replace({'GdPrv': 2, 'GdWo': 2, 'MnPrv': 1, 'MnWw': 1, 'NoFence': 0})
df['Utilities_num'] = df.Utilities.replace({'AllPub': 1, 'NoSewr': 0, 'NoSeWa': 0, 'ELO': 0})


# some maybe newer---------------------------------------------------------------------------------------------------------
df['newer_dwelling'] = df.MSSubClass.replace({20: 1, 30: 0, 40: 0, 45: 0, 50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0, 90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})

# MSSubClass transform using dict ---------------------------------------------------------------------------------------------------------
map_MSS = {x: 'Subclass_'+str(x) for x in df.MSSubClass.unique()}
print(map_MSS)
df['MSSubClass'].replace(map_MSS)

# it maybe helpful to transform some quality style features into binary 0 and 1 ---------------------------------------------------------
quality_feats = ['OverallQual', 'OverallCond', 'ExterQual_num', 'ExterCond_num', 'BsmtCond_num','GarageQual_num', 'GarageCond_num', 'KitchenQual_num']
toDrop_feats = toDrop_feats + quality_feats

all_data = df

#构造新特征 overall_poor_qu
overall_poor_qu = all_data.OverallQual.copy()
overall_poor_qu = 5 - overall_poor_qu
overall_poor_qu[overall_poor_qu<0] = 0
overall_poor_qu.name = 'overall_poor_qu'

#构造新特征 overall_good_qu
overall_good_qu = all_data.OverallQual.copy()
overall_good_qu = overall_good_qu - 5
overall_good_qu[overall_good_qu<0] = 0
overall_good_qu.name = 'overall_good_qu'

#构造新特征 overall_poor_cond
overall_poor_cond = all_data.OverallCond.copy()
overall_poor_cond = 5 - overall_poor_cond
overall_poor_cond[overall_poor_cond<0] = 0
overall_poor_cond.name = 'overall_poor_cond'

#构造新特征 overall_good_cond
overall_good_cond = all_data.OverallCond.copy()
overall_good_cond = overall_good_cond - 5
overall_good_cond[overall_good_cond<0] = 0
overall_good_cond.name = 'overall_good_cond'

#构造新特征 exter_poor_qu
exter_poor_qu = all_data.ExterQual_num.copy()
exter_poor_qu[exter_poor_qu<3] = 1
exter_poor_qu[exter_poor_qu>=3] = 0
exter_poor_qu.name = 'exter_poor_qu'

exter_good_qu = all_data.ExterQual_num.copy()
exter_good_qu[exter_good_qu<=3] = 0
exter_good_qu[exter_good_qu>3] = 1
exter_good_qu.name = 'exter_good_qu'

exter_poor_cond = all_data.ExterCond_num.copy()
exter_poor_cond[exter_poor_cond<3] = 1
exter_poor_cond[exter_poor_cond>=3] = 0
exter_poor_cond.name = 'exter_poor_cond'

exter_good_cond = all_data.ExterCond_num.copy()
exter_good_cond[exter_good_cond<=3] = 0
exter_good_cond[exter_good_cond>3] = 1
exter_good_cond.name = 'exter_good_cond'

bsmt_poor_cond = all_data.BsmtCond_num.copy()
bsmt_poor_cond[bsmt_poor_cond<3] = 1
bsmt_poor_cond[bsmt_poor_cond>=3] = 0
bsmt_poor_cond.name = 'bsmt_poor_cond'

bsmt_good_cond = all_data.BsmtCond_num.copy()
bsmt_good_cond[bsmt_good_cond<=3] = 0
bsmt_good_cond[bsmt_good_cond>3] = 1
bsmt_good_cond.name = 'bsmt_good_cond'

garage_poor_qu = all_data.GarageQual_num.copy()
garage_poor_qu[garage_poor_qu<3] = 1
garage_poor_qu[garage_poor_qu>=3] = 0
garage_poor_qu.name = 'garage_poor_qu'

garage_good_qu = all_data.GarageQual_num.copy()
garage_good_qu[garage_good_qu<=3] = 0
garage_good_qu[garage_good_qu>3] = 1
garage_good_qu.name = 'garage_good_qu'

garage_poor_cond = all_data.GarageCond_num.copy()
garage_poor_cond[garage_poor_cond<3] = 1
garage_poor_cond[garage_poor_cond>=3] = 0
garage_poor_cond.name = 'garage_poor_cond'

garage_good_cond = all_data.GarageCond_num.copy()
garage_good_cond[garage_good_cond<=3] = 0
garage_good_cond[garage_good_cond>3] = 1
garage_good_cond.name = 'garage_good_cond'

kitchen_poor_qu = all_data.KitchenQual_num.copy()
kitchen_poor_qu[kitchen_poor_qu<3] = 1
kitchen_poor_qu[kitchen_poor_qu>=3] = 0
kitchen_poor_qu.name = 'kitchen_poor_qu'

kitchen_good_qu = all_data.KitchenQual_num.copy()
kitchen_good_qu[kitchen_good_qu<=3] = 0
kitchen_good_qu[kitchen_good_qu>3] = 1
kitchen_good_qu.name = 'kitchen_good_qu'

df_qual = pd.concat((overall_poor_qu, overall_good_qu, overall_poor_cond, overall_good_cond, exter_poor_qu,
exter_good_qu, exter_poor_cond, exter_good_cond, bsmt_poor_cond, bsmt_good_cond, garage_poor_qu,garage_good_qu, 
garage_poor_cond, garage_good_cond, kitchen_poor_qu, kitchen_good_qu), axis=1)

df=pd.concat((df,df_qual),axis=1)
print('-------------------------------------------------------------------------------')
# for some categorical features, certain levels may imply better quality---------------------------------------------------------------
toDrop_feats = toDrop_feats + ['MasVnrType', 'SaleCondition', 'Neighborhood']

# 构造新特征 MasVnrType_Any
map_Mas = {'BrkCmn': 1, 'BrkFace': 1, 'CBlock': 1, 'Stone': 1, 'None': 0}
MasVnrType_Any = all_data.MasVnrType.replace(map_Mas)
MasVnrType_Any.name = 'MasVnrType_Any'

# 构造新特征 SaleCondition_PriceDown
map_Sale = {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0}
SaleCondition_PriceDown = all_data.SaleCondition.replace(map_Sale)
SaleCondition_PriceDown.name = 'SaleCondition_PriceDown'

# 构造新特征 Neighborhood_good
neigh_good_feats = ['NridgHt', 'Crawfor', 'StoneBr', 'Somerst', 'NoRidge']
df['Neighborhood_good'] = 0
df.loc[df.Neighborhood.isin(neigh_good_feats), 'Neighborhood_good'] = 1

df=pd.concat((df,MasVnrType_Any,SaleCondition_PriceDown),axis=1)
print(df.columns)

# Monthes with the lagest number of deals may be significant--------------------------------------------------------------------------
df['season'] = df.MoSold.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4})

# Numer month is not significant, it maybe helpful to transform them into object feature------------------------------------------------
map_Mo = {1: 'Yan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Avg', 9: 'Sep', 10: 'Oct',11: 'Nov', 12: 'Dec'}
df = df.replace({'MoSold': map_Mo})

# some features about years---------------------------------------------------------------------------------------------------------------

# sold at the same year as bulit
df['SoldImmediate'] = 0
df.loc[(df.YrSold == df.YearBuilt), 'SoldImmediate'] = 1

# reconstructed since first built
df['Recon'] = 0
df.loc[(df.YearBuilt < df.YearRemodAdd), 'Recon'] = 1

# reconstructed after sold
df['ReconAfterSold'] = 0
df.loc[(df.YrSold < df.YearRemodAdd), 'ReconAfterSold'] = 1

# reconstructed the same year as sold
df['ReconEqualSold'] = 0
df.loc[(df.YrSold == df.YearRemodAdd), 'ReconEqualSold'] = 1

# Years are too much, it maybe helpful to devide them into groups, and delete the original ones-----------------------------------------

year_map=pd.concat(pd.Series('YearGroup'+str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0,7))

df.YearBuilt = df.YearBuilt.map(year_map)
df.YearRemodAdd = df.YearRemodAdd.map(year_map)
df.GarageYrBlt = df.GarageYrBlt.map(year_map)

df.loc[df.GarageYrBlt.isnull(),'GarageYrBlt']='NoGarage'
print('---------------------------------------------------------------------')

print(len(toDrop_feats),df.shape)

# 通过训练 svc 模型来构造新特征 price_category---------------------------------------------------------------------------------------------
pc=pd.Series(np.zeros(d_train.shape[0]))

pc[:]='pc1'
pc[d_train.SalePrice>=150000]='pc2'
pc[d_train.SalePrice>=220000]='pc3'

columns_for_pc=['Exterior1st', 'Exterior2nd', 'RoofMatl', 'Condition1', 'Condition2', 'BldgType']
toDrop_feats=toDrop_feats+columns_for_pc

X_t=pd.get_dummies(d_train.loc[:,columns_for_pc])

from sklearn.svm import SVC
svm=SVC(C=100)
svm.fit(X_t,pc)

X_t=pd.get_dummies(df.loc[:,columns_for_pc])
pc_pred=pd.DataFrame(svm.predict(X_t),columns=['price_categroy'],index=df.index)

df=pd.concat((df,pc_pred),axis=1)
df.loc[df.price_categroy=='pc1','price_categroy']=1
df.loc[df.price_categroy=='pc2','price_categroy']=2
df.loc[df.price_categroy=='pc3','price_categroy']=3

print('-------------------------------------------------------------------------------------------')
#对数值型特征的取值进行规约，归一化
numeric_feat=df.dtypes[df.dtypes != 'object'].index 
t=df[numeric_feat].quantile(.95)

use_max_scater=t[t==0].index
use_95_scater=t[t!=0].index

df[use_max_scater] = df[use_max_scater]/df[use_max_scater].max()
df[use_95_scater] = df[use_95_scater]/df[use_95_scater].quantile(.95)

print('------------------------------------------------------------------------------------------------')

#对数值型的特征进行分布参数计算，对必要的特征进行取对数化

numeric_feat=df.dtypes[df.dtypes != 'object'].index
from scipy.stats import skew
feat_skew=df[numeric_feat].apply(lambda x:skew(x.dropna()))
skewed_num_feats=feat_skew[feat_skew>.75].index

df[skewed_num_feats]=np.log1p(df[skewed_num_feats])
print('----------------------------------------------------------------')


# 对特征进行二元特征抽取------------------------------------------------------------------------------------
df_new=pd.get_dummies(df)
print(df_new.shape)
print(df_new.columns)
print('-----------------------------------')



#构造交叉项特征----------------------------------------------------------------
from itertools import product, chain
def poly(X):
	areas = ['LotArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'BsmtUnfSF']
	t = chain(df_qual.axes[1].get_values(),
	['OverallQual_num', 'OverallCond_num', 'ExterQual_num', 'ExterCond_num', 'BsmtCond_num', \
	'GarageQual_num', 'GarageCond_num', 'KitchenQual_num', 'HeatingQC_num', \
	'MasVnrType_Any', 'SaleCondition_PriceDown', 'Recon',
	'ReconAfterSold', 'SoldImmediate'])
	for a, t in product(areas, t):
		x = X.loc[:, [a, t]].prod(1)
		x.name = a + '_' + t
		yield x
print('poly:',poly(df_new))

XP = pd.concat(poly(df_new), axis=1)
df_new = pd.concat((df_new, XP), axis=1)
print(df_new.shape)

#构造相加特征项-------------------------------------------------------------------------
df_new['bsmtfinsf_1_2_sum']=df_new['BsmtFinSF1']+df_new['BsmtFinSF2']

df_new['sum_FlrSF']=df_new['1stFlrSF']+df_new['2ndFlrSF']


#################################### prepare x,y ################################################################
df_final = df_new.copy()
print(df_final.shape[0])
X_train = df_final[:d_train.shape[0]]
print(X_train.shape)
y_train = np.log(d_train.SalePrice) #对 price 取对数
y_train.columns=['price']
print(y_train.shape)
X_test = df_final[d_train.shape[0]:]
print(X_test.shape)

#删除两个异常值
outliers_id = np.array([524, 1299])
outliers_id = outliers_id - 1 # id starts with 1, index starts with 0
X_train = X_train.drop(outliers_id, axis=0)
y_train = y_train.drop(outliers_id, axis=0)

X_train.to_csv('E:\\house\\x_trainf.csv',index=None)
y_train.to_csv('E:\\house\\y_trainf.csv',index=None)
X_test.to_csv('E:\\house\\x_testf.csv',index=None)
