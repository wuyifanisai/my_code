#-*- coding: utf-8 -*-
#使用K-Means算法聚类消费行为特征数据

import pandas as pd

#参数初始化
inputfile = 'E:\k.xls' #销量及其他属性数据
outputfile = 'E:\k2.xls' #保存结果的文件名
k = 3 #聚类的类别
iteration = 500 #聚类最大循环次数
data = pd.read_excel(inputfile, index_col = 'Id') #读取数据
data_zs = 1.0*(data - data.mean())/data.std() #数据标准化

from sklearn.cluster import KMeans
model = KMeans(n_clusters = k, max_iter = iteration) #分为k类，并发数4
model.fit(data_zs) 


#简单打印结果
r1 = pd.Series(model.labels_).value_counts() #统计各个类别的数目
r2 = pd.DataFrame(model.cluster_centers_) #找出聚类中心

r = pd.concat([r2, r1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
r.columns = list(data.columns) + [u'类别数目'] #重命名表头

#详细输出原始数据及其类别
r = pd.concat([data, pd.Series(model.labels_, index = data.index)], axis = 1)  #详细输出每个样本对应的类别
r.columns = list(data.columns) + [u'聚类类别'] #重命名表头
r.to_excel(outputfile) #保存结果

#plot the density figure
def density_plot(data,title):
	import matplotlib.pyplot as plt
	plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
	plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
	for i in range(len(data.iloc[0])):
		plt.figure()
		(data.iloc[:,i]).plot(kind='kde',label=data.columns[i],linewidth=2)
		plt.ylabel(u'density')
		plt.xlabel(u'num')
		plt.title(u'聚类类别%s'%title)
		plt.legend()
	plt.show()



#comparsion of data
def density_compare_plot(data):
	import matplotlib.pyplot as plt
	plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
	plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
	plt.figure()
	for j in range(len(data.iloc[0])-1):
		for l in range(k):
			((data[r[u'聚类类别']==l]).iloc[:,j]).plot(kind='kde',label='club_%s'%(l+1),linewidth=2*(j+1))
			plt.ylabel(u'density')
			plt.xlabel(u'num')	
			if j==0:
				s='R'
			if j==1:
				s='F'
			if j==2:
				s='M'
			plt.title(u'比较属性%s'%(s))
			plt.legend()
		plt.show()
		




print('**************************************************')

#density_compare_plot(r)
print(r)
'''
def density_plot(data): #自定义作图函数
  import matplotlib.pyplot as plt
  plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
  plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
  p = data.plot(kind='kde', linewidth = 2, subplots = True, sharex = False)
  [p[i].set_ylabel(u'密度') for i in range(k)]
  plt.legend()
  return plt

pic_output = '../tmp/pd_' #概率密度图文件名前缀
for i in range(k):
  density_plot(data[r[u'聚类类别']==i]).savefig(u'%s%s.png' %(pic_output, i))
'''


from sklearn.manifold import TSNE

tsne = TSNE()
tsne.fit_transform(data_zs) #进行数据降维
tsne = pd.DataFrame(tsne.embedding_, index = data_zs.index) #转换数据格式


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
print(tsne)

#不同类别用不同颜色和样式绘图
d = tsne[r[u'聚类类别'] == 0]
plt.plot(d[0], d[1], 'r.')
d = tsne[r[u'聚类类别'] == 1]
plt.plot(d[0], d[1], 'go')
d = tsne[r[u'聚类类别'] == 2]
plt.plot(d[0], d[1], 'b*')
plt.show()
