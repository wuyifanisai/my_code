# -*- coding: utf-8 -*-'
#对美国历年的国情咨文的文本的挖掘，分析历年总统的政治思想特征
import re
import pytagcloud
import jieba #导入结巴分词，需要自行下载安装
import random
import numpy as np
import pandas as pd
import math
import gensim 
from gensim import corpora,models
from gensim.models import word2vec
import logging 
print('#######################文本导入并分词######################################')
#停用词汇
word_not_need_list=['我们','你们','他们','一个','的','了','一种','一起','掌声','人们',
'为了','全文','知道','发现','我国','不会','任何','所有','那些','然而','如果','起来',
'这个','这场','没有','看到','永远','自己','得到','必须','但是','需要','工作','目标',
'继续','一位','帮助','已经','取得','无法','现在','方面','时刻','获得','而且','就是',
'一些','当中','通过','能够','这项','唯一','成为','不是','因此','作出','这种','带来',
'进行','以及','一项','因为','这是','今晚','以往','这一','如此','之一','']#存放高频出现但又没有价值的词汇，例如介词等等
#将txt文本的自动换行取消！！！！

for i in range(1):# 每一次只处理一年的国情咨文的文本数据

	############################################################################################################################################
	###要通过gensim.models 的 word2vec模型进行文本词汇的词向量构造以及分析文本主题词，	
	#需要将预料文本转换文一个一个句子，那个句子的词汇拆分后放入一个列表中，一个句子对应一个列表，该句子中的每个词语用放在列表中。
	#最后所有句子的列表放在一个总的列表中，例如下面的例子
	#例如#sentences =[['first','dog','fast'],['second','dog'],['first','dog']] ，获取了一段文本中的三个句子

	sentences=[]#存放所有句子的总列表
	txtpath='E:\\Master\\PPDAMcode\\PROJECT\\President_speech\\%d.txt'%(2010)
	fp=open(txtpath,'r',encoding= 'gbk')
	for lines in fp.readlines():       #fp.readlines是读取txt中所有的文本内容，lines是从中拿出一行句子
		lines=lines.replace('\n','')   #每一句子的换行符号消除掉
		lines=lines.split(' ')        #对每一行通过空格进行截断，分成若干个不同的语句，放在不同的列表中，若没有空格，则整行句子就放在一个列表中
		print(lines)

		#对每个列表中的每个句子进行分词
		sentence=[]
		for word in jieba.cut(lines[0]):
			#对每一个分出来的词汇进行判断，除掉符号标点等等不需要的内容
			if re.match(u"[\u4e00-\u9fa5]{2,4}",word):  #判断是否是2到4个字的词汇
				if word in word_not_need_list:          #判断分出来的词汇是否为不需要的词汇
					continue
				else:
					sentence.append(word)
		#print(sentence)
		sentences.append(sentence)
	print(sentences)

	############################################################################################################################
	print('***********************')
	################################输出这段文本(包含所有句子列表)的主题词##################
	film_dict = corpora.Dictionary(sentences)

	i=0
	for w in film_dict.values():
		i+=1
		print(i,w)
	
	film_corpus = [film_dict.doc2bow(i) for i in sentences]
	print(film_corpus)
	#
	film_lda = models.LdaModel(film_corpus, num_topics = 3, id2word = film_dict)
	for i in range(3):
  		print(film_lda.print_topic(i)) #输出每个主题

  	########################################################################################
	

	####################################利用Word2Vec模型进行词向量化计算，计算词汇相关程度#############################################################################
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	#sentences = word2vec.Text8Corpus(u"E:\word.txt",encoding= 'utf-8')
	w=[]
	for i in range(100):
		model = gensim.models.Word2Vec(sentences, size=200)  # 训练skip-gram模型; 默认window=5
		
		print(model.similarity(u'美国',u'经济'))# 计算两个词的相似度/相关程度
		print(model.most_similar(u'家庭'))# 计算某个词的相关词列表
		
		y=model.most_similar(u'美国',topn=10)
		for item in y:
			print(item[0],item[1])
			w.append([item[0],item[1]])
		c=1/0
		w_min=1
		w_max=0
		for i in range(len(w)):
			if w_min>w[i][1]:
				w_min=w[i][1]

		for i in range(len(w)):
			if w_max<w[i][1]:
				w_max=w[i][1]

		w1=[]
		w2=[]
		w3=[]

		for i in range(len(w)):
			if w[i][1]>(w_min+(w_max-w_min)*2/3):
				if w[i][0] not in w1:
					w1.append(w[i][0])
			if (w_min+(w_max-w_min)*1/3)<w[i][1]<(w_min+(w_max-w_min)*2/3):
				if w[i][0] not in w2:
					w2.append(w[i][0])
			if (w_min)<w[i][1]<(w_min+(w_max-w_min)*1/3):
				if w[i][0] not in w3:
					w3.append(w[i][0])

	print('与关键词美国有关的','该年度第一重复词汇----->>>>',w1)
	print('与关键词美国有关的','该年度第二重复词汇----->>>>',w2)
	print('与关键词美国有关的','该年度第三重复词汇----->>>>',w3)

	###############################################制作标签云############################################################
	count=[]

	for i in range(len(sentences)):
		for j in range(len(sentences[i])):
			count.append(sentences[i][j])

	from collections import Counter
	counts=Counter(count).items()
	#print(counts)

	'''
	from pytagcloud import create_tag_image,make_tags
	tags=make_tags(counts,maxsize=100)
	#create_tag_image(tags,'word.png',size=(500,500),fontname='MicrosoftYaHei',background=(0, 0, 0))
	'''

