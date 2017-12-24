#-*- coding: utf-8 -*-
#数据规范化
import pandas as pd
import numpy as np
datafile = 'E:\juhua.xls' #参数初始化
data = pd.read_excel(datafile,index_col=None, header = 0) #读取数据
#如果需要的花，可以将一个数据表pandas中的所有数据一起进行规范化，或者单独对某一属性的数据进行输入规范化处理

#data=(data - data.min())/(data.max() - data.min()) #最小-最大规范化

#data=(data - data.mean())/data.std() #零-均值规范化

data=data/10**np.ceil(np.log10(data.abs().max())) #小数定标规范化

print(data)