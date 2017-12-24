import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('E:\\Breast Cancer Wisconsin (Diagnostic) Data Set\\data.csv')
print(df.columns)
print('---------------------------')

df.loc[df.diagnosis=='B','diagnosis']=np.float64(0.0)
df.loc[df.diagnosis=='M','diagnosis']=np.float64(1.0)

df=df.loc[:,'diagnosis':'fractal_dimension_worst']

'''
# 利用单变量特征与diagnosis特征作散点图，逐个考察每个特征与diagnosis的关系
def scatterplot(var):
	sns.pairplot(df[['diagnosis']+[var]])
	plt.show()

scatterplot('area_se')
# 利用箱形图来观察某一个特征，在 diagnosis 取 0,1时候该特征的象形图分布
def boxplot(var):
	data = pd.concat([df['diagnosis'], df[var]], axis=1)
	f, ax = plt.subplots(figsize=(8, 6))
	fig = sns.boxplot(x='diagnosis', y=var, data=data)
	fig.axis(ymin=df[var].min(), ymax=df[var].max())
	plt.show()
boxplot('concavity_mean')
'''

# 计算每个特征对于目标变量的区分度

feat_discrimination={} # key:discrimination ,value:feat
best_feats_discrimination=[0,0,0,0,0] #存放5个区分度最高的特征
for feat in df.columns:

	best_feats_discrimination.sort()
	best_feats_discrimination.reverse()

	if feat != 'diagnosis':
		a=df.loc[df.diagnosis == 0,feat].mean()
		b=df.loc[df.diagnosis == 1,feat].mean()
		feat_discrimination[abs(a-b)/max(a,b)] = feat

		if abs(a-b)/max(a,b)>best_feats_discrimination[-1]:
			best_feats_discrimination[-1] = abs(a-b)/max(a,b)
print('区分度最高的5个特征：')
best_feat=[]
for n in best_feats_discrimination:
	print(feat_discrimination[n])
	best_feat.append(feat_discrimination[n])


# 各个特征的区分度柱状图表示
a=pd.Series(list(feat_discrimination.keys()))
b=pd.Series(list(feat_discrimination.values()))
dis=pd.DataFrame({'feat':b,'discrimination':a})
dis=dis.sort_index(by='discrimination',ascending=False)
dis.index=list(dis['feat'])
dis.plot(kind='bar')
plt.show()



# 除了分类的标签外，可以对特征之间的相关性进行分析
corrmat = df.loc[:,best_feat].corr()
k = 5 #number of variables for heatmap
cols = corrmat.nlargest(k, 'concavity_mean')['concavity_mean'].index
#cols = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond','SalePrice']
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols, xticklabels=cols)
plt.show()




