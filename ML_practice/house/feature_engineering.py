import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('E:\\house\\df.csv')
df=df.loc[:,'MSSubClass':'SaleCondition']
print(df)

#######################################  进行特征工程  ##########################################

#以下是obj特征中经常出现的字符型特征
ordinal_words = ['Ex', 'Gd', 'TA', 'Fa', 'Po']

for x in 