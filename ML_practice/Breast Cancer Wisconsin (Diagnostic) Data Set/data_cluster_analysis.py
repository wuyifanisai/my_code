import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.externals import joblib


label=pd.read_csv('E:\\Breast Cancer Wisconsin (Diagnostic) Data Set\\data.csv')
label=label.iloc[:,1]
print(label.value_counts())

df=pd.read_csv('E:\\Breast Cancer Wisconsin (Diagnostic) Data Set\\data_1.csv')
df=df.loc[:,['concavity_mean','area_se','concave points_mean','concavity_worst','area_worst']]

print(df)

'''
for k in range(1,5):
	model=KMeans(n_clusters=k,n_init=20,max_iter=500)
	model.fit(df)
	#print('center of every cluster:',model.cluster_centers_)
	print('k=',k,'  聚类的效果指标：',model.inertia_)

model=KMeans(n_clusters=3,n_init=50,max_iter=500)
model.fit(df)



#简单打印结果
r1 = pd.Series(model.labels_).value_counts() #统计各个类别的数目
r2 = pd.DataFrame(model.cluster_centers_) #找出聚类中心

clusters = pd.concat([r2, r1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
clusters.columns = list(df.columns) + [u'类别数目'] #重命名表头

#详细输出原始数据及其类别
r = pd.concat([df, pd.Series(model.labels_, index = df.index)], axis = 1)  #详细输出每个样本对应的类别
r.columns = list(df.columns) + [u'聚类类别'] #重命名表头


#聚类可视化--------------------------------
from sklearn.manifold import TSNE

tsne = TSNE()
tsne.fit_transform(df) #进行数据降维
print(tsne.embedding_)
tsne = pd.DataFrame(tsne.embedding_, index = df.index) #转换数据格式

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


#不同类别用不同颜色和样式绘图
d = tsne[r[u'聚类类别'] == 0]
plt.plot(d[0], d[1], 'r*')

d = tsne[r[u'聚类类别'] == 1]
plt.plot(d[0], d[1], 'g*')

d = tsne[r[u'聚类类别'] == 2]
plt.plot(d[0], d[1], 'b*')

plt.show()



# 经过聚类分析发现，除了某一个类别以外，其余的类别都能够以较高的概率判断出样本的标签类别 ！！！！！
for i in range(3):
	print('label of sample in cluster',i)
	cluster_index=list(r[r[u'聚类类别'] == i].index)
	print(label[cluster_index].value_counts())

dd=pd.DataFrame({
		u'0':label.loc[list(r[r[u'聚类类别'] == 0].index)].value_counts(),
		u'1':label.loc[list(r[r[u'聚类类别'] == 1].index)].value_counts(),
		u'2':label.loc[list(r[r[u'聚类类别'] == 2].index)].value_counts()
		})
print(dd)
(dd.T).plot(kind='bar',stacked=True)
plt.show()


#保存模型
joblib.dump(model,'E:\\Breast Cancer Wisconsin (Diagnostic) Data Set\\km.model')


#加载模型 TEST
km=joblib.load('E:\\Breast Cancer Wisconsin (Diagnostic) Data Set\\km.model')

k=0
n=0
for i in range(len(df)):
	if km.predict(df.loc[i,['concavity_mean','area_se','concave points_mean','concavity_worst','area_worst']])[0] == 2:
		n=n+1
		if label[i] =='M':
			k=k+1
print(k/n)
'''

##############################   层次聚类分析 ##############################################
k =3 #聚类数
from sklearn.cluster import AgglomerativeClustering #导入sklearn的层次聚类函数
model = AgglomerativeClustering(n_clusters = k, linkage = 'ward')
model.fit(df) #训练模型

#详细输出原始数据及其类别
r = pd.concat([df, pd.Series(model.labels_, index = df.index)], axis = 1)  #详细输出每个样本对应的类别
r.columns = list(df.columns) + [u'聚类类别'] #重命名表头

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

style = ['ro-', 'go-', 'bo-','ro-','go-']
xlabels = ['concavity_mean','area_se','concave points_mean','concavity_worst','area_worst']


for i in range(k): #逐一作图，作出不同样式
  plt.figure()
  tmp = r[r[u'聚类类别'] == i].iloc[:,:4] #提取每一类
  for j in range(len(tmp)):
    plt.plot(range(1, 5), tmp.iloc[j], style[i])
  
  plt.xticks(range(1, 5), xlabels, rotation = 20) #坐标标签
  plt.title(u'类别%s' %(i+1)) #我们计数习惯从1开始
  plt.subplots_adjust(bottom=0.15) #调整底部
  plt.show()
  #plt.savefig(u'%s%s.png' %(pic_output, i+1)) #保存图片


for i in range(3):
	print('label of sample in cluster',i)
	cluster_index=list(r[r[u'聚类类别'] == i].index)
	print(label[cluster_index].value_counts())



'''
通过两种聚类的分析可以发现，聚类可以很大程度上为类别的标记鉴定提供很大的支持

'''

'''
#########################  density based method ####################################
from sklearn.cluster import DBSCAN
model=DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto')

model.fit(df)

r=pd.Series(model.labels_)


#聚类可视化--------------------------------
from sklearn.manifold import TSNE
tsne = TSNE()
tsne.fit_transform(df) #进行数据降维
print(tsne.embedding_)
tsne = pd.DataFrame(tsne.embedding_, index = df.index) #转换数据格式
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

#不同类别用不同颜色和样式绘图
d = tsne[r == -1]
plt.plot(d[0], d[1], 'y*')

d = tsne[r == 0]
plt.plot(d[0], d[1], 'r*')

d = tsne[r == 1]
plt.plot(d[0], d[1], 'g*')

d = tsne[r == 2]
plt.plot(d[0], d[1], 'b*')

plt.show()
'''