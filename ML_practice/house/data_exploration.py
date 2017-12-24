#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#check the given train data
df=pd.read_csv('E:\\house\\train.csv')


#print(df)
'''
#check the index (number) of train data
print(df.index)
print('-------------------------')

#check the index of train data
print(df.columns)
print('-------------------------')

#check the label --- price
print('the price describe:',df['SalePrice'].describe())
df['SalePrice'].plot()
plt.show()
sns.distplot(df['SalePrice'])#数据分布直方图
plt.show()
#分布图曲线的两个参数 skew kurt 
print("Skewness: %f" % df['SalePrice'].skew())
print("Kurtosis: %f" % df['SalePrice'].kurt())
'''
print('---------------------')
#查看训练数据的特征数字类型
print(df.dtypes.value_counts())
print()
#分别查看不同类型的特征名
object_feats=list(df.dtypes[df.dtypes=='object'].index)
print('object:',object_feats)
print(len(object_feats))

int64_feats=list(df.dtypes[df.dtypes=='int64'].index)
print('int_feature:',int64_feats)
print(len(int64_feats))

float64_feats=list(df.dtypes[df.dtypes=='float64'].index)
print('float_feature:',float64_feats)
print(len(float64_feats))
print()

'''
#take a look at the num_type data--------------------------------------------------------- ----------------------------------------------------

#int64 给出int64类型的特征与price相关度的排序
int64_corr_price=df[int64_feats].corr()['SalePrice'].order()
print('int64 特征与price的相关度：',int64_corr_price)

sns.set()
cols=list(int64_corr_price[-5:].index)
sns.pairplot(df[cols])#几个相关度较强的int64特征与price的相关图
plt.show()


#float64 给出float64类型的特征与price相关度的排序
float64_corr_price=df[float64_feats+['SalePrice']].corr()['SalePrice'].order()
print('FLOAT64 特征与price的相关度：',float64_corr_price)

#take a look at the heatmap of num_type feature
corrmat = df[float64_feats+['SalePrice']].corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.4, square=True)
plt.show()

corrmat = df[int64_feats].corr()
f, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(corrmat, vmax=.4, square=False)
plt.show()
'''


def check_data(s):#给出该数值型变量的分布以及与price的散点关系图
	sns.distplot(df[s])
	sns.pairplot(df[[s]+['SalePrice']])
	plt.show()

check_data('OverallQual')


#通过箱型图来分析每一个特征的不同取值对于price的影响，一半是针对取值个数不是很多的特征------------------------------------------
def box_plot(var):
	data = pd.concat([df['SalePrice'], df[var]], axis=1)
	f, ax = plt.subplots(figsize=(8, 6))
	fig = sns.boxplot(x=var, y="SalePrice", data=data)
	fig.axis(ymin=0, ymax=800000)
	plt.show()
#box_plot('MSZoning')



'''
#数值型相关性矩阵表示,选择与price相关性最大的几个数值型特征来计算与price的相关具体值矩阵-------------------------------------------------------------------------------------------------
#saleprice correlation matrix
corrmat = df[int64_feats].corr()
#corrmat = df[float64_feats+['SalePrice']].corr()
k = 5 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
#cols = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond','SalePrice']
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols, xticklabels=cols)
plt.show()
'''

#检验某一数据的分布规范性，是否分布正常，符合正态分布等假定
def check_dis_normal(s,flag):
	sns.distplot(df[s])
	plt.show()
	stats.probplot(df[s], plot=plt)
	plt.show()
	if flag: #通过对数处理，将原始分布不规范的数据转换为分布规范的数据
		sns.distplot(np.log(df[s]))
		plt.show()
		stats.probplot(np.log(df[s]), plot=plt)
		plt.show()
check_dis_normal('SalePrice',1)

#考察某一个字符型特征的分布以及与price分类的关系
def object_feat_check(s):

	#p_c=df['SalePrice']//200000
	#d=pd.concat((df[s],p_c),axis=1)
	#d.columns=['object_feat','p_c']

	d=pd.DataFrame(df[s])
	d.columns=['object_feat']
	dd=pd.DataFrame({
		u'0':d.object_feat[df['SalePrice']<100000].value_counts(),
		u'1':d.object_feat[df['SalePrice']<170000].value_counts(),
		u'2':d.object_feat[df['SalePrice']<300000].value_counts(),
		u'3':d.object_feat[df['SalePrice']>=300000].value_counts(),
		})
	print(dd)
	print('----------')
	dd.plot(kind='bar', stacked=False)
	plt.title(s)
	plt.show()
object_feat_check('Electrical')

# MAKE A SCATTER 画出price 与 某一个 变量之间的散点图
def make_scatter(s):
	plt.scatter(df[s],df['SalePrice']) 
	plt.show()    
#make_scatter('LotFrontage')


'''
#object_feature value_count()
def value_count(s):
	print(s,':')
	print(df[s].value_counts())
for s in object_feats:
	value_count(s)
	print()
'''

for s in int64_feats:
	sns.distplot(df[s])
	plt.show()
	
	