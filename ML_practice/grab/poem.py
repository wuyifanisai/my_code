#coding: utf-8
import urllib.request
import re
import pandas as pd
import jieba #导入结巴分词，需要自行下载安装

""" ***************使用正则表达式提取网页中的标题、链接、图片***************"""
mycut = lambda s: ' '.join(jieba.cut(s)) #自定义简单分词函数
web=['http://www.lz13.cn/shiju/72008.html',
'http://www.lz13.cn/shiju/72006.html',
'http://www.lz13.cn/shiju/72008.html',
'http://www.lz13.cn/mingrenmingyan/50627.html',
'http://www.lz13.cn/mingrenmingyan/50188.html',
'http://www.lz13.cn/shiju/72004.html',
'http://www.lz13.cn/shiju/132602.html',
'http://www.lz13.cn/shiju/136708.html'
]

word=[]
for j in range(len(web)):
	Target=web[j]
	url=urllib.request.urlopen(Target)#urlopen返回 一个类文件对象
	page=url.read()#读取文件内容至pager
	url.close()
	page=page.decode('utf-8')#findall要求的对象格式为str
	page_poem=re.compile('</p><p>(.+?)&mdash;&mdash')
	page_p=re.compile('、(.+?)。')
	k=0
	for data in page_p.findall(page):
		w=[]
		p=[]
		#print(data)
		d=mycut(data).strip()
		p=d.split(' ')
		k=k+1
		for i in p:
			pp=re.compile('[\u4e00-\u9fa5]+')
			str=i
			if pp.findall(str):
				w.append(str)
			else:
				p.remove(str)
		if j==4:
			for i in range(500):
				word.append(w)
		else:
			word.append(w)
		#print(p)
print(len(word))

'''
from collections import Counter
counts=Counter(word).items()
print(counts)

from pytagcloud import create_tag_image,make_tags
tags=make_tags(counts,maxsize=100)
create_tag_image(tags,'songci1.png',size=(1000,1000),fontname='MicrosoftYaHei',background=(0, 0, 0))
'''

import gensim
from gensim.models import word2vec
import logging 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#sentences =[['first','dog','fast'],['second','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog']] 
sentences=word
sentences = word2vec.Text8Corpus(u"E:\word.txt",encoding= 'utf-8')
model = gensim.models.Word2Vec(sentences, size=200)  # 训练skip-gram模型; 默认window=5
###########################use the model##########################################################
#print(model.similarity(u'月',u'思'))# 计算两个词的相似度/相关程度
print(model.most_similar(u'月'))# 计算某个词的相关词列表



y=model.most_similar(u'恨',topn=10)
print('most similarity words with 恨')
for item in y:
	print(item[0],'---->',item[1])
	print('\n')


'''
# 寻找对应关系
y3 = model.most_similar([u'冯小刚', u'电影'], [u'范冰冰'], topn=3)
for item in y3:
    print(item[0], item[1])
print ("--------\n")



# 寻找不合群的词
y4 = model.doesnt_match(u"潘金莲 范冰冰 冯小刚".split())
print (u"不合群的词：", y4)
print("--------\n")
'''


'''
# 保存模型，以便重用
model.save(u"w.model")
# 对应的加载方式
model_2 = Word2Vec.Word2Vec.load("w.model")
'''
