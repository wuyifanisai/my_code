# -*- coding: utf-8 -*-'
import numpy as np
from numpy import mat
def Jaccard(a, b): #只能0-1向量ARRAY，进行的是向量运算
  return 1.0*(a*b).sum()/(a+b-a*b).sum()

def COS_sim(a,b):  #数据应该是一维矩阵MATRIX，方法中进行的是矩阵运算
	num=float(a*b.T)
	denom=np.linalg.norm(a)*np.linalg.norm(b)
	return 0.5+0.5*(num/denom)

'''
a=mat([[1,0,0,1,0]])
b=mat([[1,0,0,1,0]])
c=mat([[1,0,1,0,0]])
print(COS_sim(a,b))
print(COS_sim(a,c))
'''
#计算两个物品之间的相似度是基于每一个物品的用户喜爱程度向量，向量中的第n的数值表示第n个用户对该物品的喜爱程度量化值
#通过比较两个物品各自的用户喜爱程度向量，可以计算两个物品之间的相似度
class Recommender():
  
  sim = None #ÏàËÆ¶È¾ØÕó
  
  def similarity(self, x, distance): #¼ÆËãÏàËÆ¶È¾ØÕóµÄº¯Êý
    y = np.ones((len(x), len(x)))
    for i in range(len(x)):
      for j in range(len(x)):
        y[i,j] = distance(x[i], x[j])
    return y
  
  def fit(self, x, distance = COS_sim): #ÑµÁ·º¯Êý
    self.sim = self.similarity(x, distance)
  
  def recommend(self, a): #ÍÆ¼öº¯Êý
    return np.dot(self.sim, a)*(1-a)



print('**********************************')
 

datamat=mat([[2,0,0,4,4,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,5],[0,0,0,0,0,0,0,1,0,4,0],[3,3,4,0,3,0,0,2,2,0,0],[5,5,5,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,5,0,0,5,0],[4,0,4,0,0,0,0,0,0,0,5],[0,0,0,0,0,4,0,0,0,0,4],[0,0,0,0,0,0,5,0,0,5,0],[0,0,0,3,0,0,0,0,4,5,0],[1,1,2,1,1,2,1,0,4,5,0]])
print(datamat)
#datamat=np.array(datamat)
'''
u,sigma,v=np.linalg.svd(datamat)
print(sigma)
datamat_1=datamat[:,:9]
print(datamat_1)
'''
user_vec0=np.array([0,0,0,3,4,0,0,0,0,0,0])#原始用户使用物品情况向量
user_vec=np.array([0,0,0,1,1,0,0,0,0,0,0])#用户使用物品的0-1向量
R=Recommender()
R.fit(datamat)#由于方法中相似度计算使用多维向量运算，所以输入的用户-物品矩阵其实是多维向量数据形式
#print(mat(R.sim))
s=(mat(R.sim)*mat(user_vec0).T)#先使用原始用户使用物品情况向量计算所有物品针对该用户已使用物品的相似度综合

#print(s)
print(np.multiply(s,mat(1-user_vec).T))#最后使用用户使用物品的0-1向量来筛选出该用户未使用过的物品的相似度综合，进行排名然后推荐
