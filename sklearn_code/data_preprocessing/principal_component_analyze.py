#-*- coding: utf-8 -*-
#主成分分析 降维
import pandas as pd
import numpy as np

#参数初始化
inputfile = '../data/principal_component.xls'
outputfile = '../tmp/dimention_reducted.xls' #降维后的数据

data = pd.read_excel(r'E:\Master\PPDAMcode\AIR_project\hangzhou2014.xls', header = None) #读入数据
print(data)
from sklearn.decomposition import PCA

pca = PCA(n_components=3) 
pca.fit(data)
print(pca.components_) #返回模型的各个特征向量
print(pca.explained_variance_ratio_) #返回各个成分各自的方差百分比
print(pd.DataFrame(pca.fit_transform(data)))
