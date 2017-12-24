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
label=df['diagnosis']

label[label=='B']=0
label[label=='M']=1

print(list(df[label==0].index))
c=1/0

print('---------------------------')

df.loc[df.diagnosis=='B','diagnosis']=np.float64(0.0)
df.loc[df.diagnosis=='M','diagnosis']=np.float64(1.0)

df=df.loc[:,'radius_mean':'fractal_dimension_worst']

print(df.head(1))

# check missing value 
nan=df.isnull().sum(axis=0)
print('feature containing missing point :')
print(nan[nan>0])
# there is no missing point


# 检查每个特征的数据分步偏度 skew 
from scipy.stats import skew
feat_skew=df.apply(lambda x:skew(x.dropna()))
print('feat whose skew >0.75:')
print(list(feat_skew[feat_skew>0.75].index))

# make a log transform for feat whose skew > 0.75
df.loc[:,list(feat_skew[feat_skew>0.75].index)] = np.log1p(df.loc[:,list(feat_skew[feat_skew>0.75].index)])

# 数据归一化（去平均，除掉标准差）
df=(df-df.mean())/df.std()
'''
for f in df.columns:
	sns.distplot(df[f])
	plt.title('f')
	plt.show()
'''
print(df.shape)

df.to_csv('E:\\Breast Cancer Wisconsin (Diagnostic) Data Set\\data_1.csv',index=None)
label.to_csv('E:\\Breast Cancer Wisconsin (Diagnostic) Data Set\\label.csv',index=None)