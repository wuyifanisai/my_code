#coding: utf-8
import urllib.request
import re
import pandas as pd
import jieba #导入结巴分词，需要自行下载安装

""" ***************使用正则表达式提取网页中的标题、链接、图片***************"""
mycut = lambda s: ' '.join(jieba.cut(s)) #自定义简单分词函数
web=['https://movie.douban.com/subject/26630781/comments?sort=new_score',
'https://movie.douban.com/subject/26630781/comments?start=24&limit=20&sort=new_score',
'https://movie.douban.com/subject/26630781/comments?start=48&limit=20&sort=new_score',
'https://movie.douban.com/subject/26630781/comments?start=74&limit=20&sort=new_score',
'https://movie.douban.com/subject/26630781/comments?start=105&limit=20&sort=new_score',
'https://movie.douban.com/subject/26630781/comments?start=130&limit=20&sort=new_score',
'https://movie.douban.com/subject/26630781/comments?start=152&limit=20&sort=new_score',
'https://movie.douban.com/subject/26630781/comments?start=174&limit=20&sort=new_score',
'https://movie.douban.com/subject/26630781/comments?start=196&limit=20&sort=new_score']

print(len(web))
k=0
n=0
word=[]



'''
for i in range(len(web)):
########################################################################################
	n=n+1
	Target=web[i]
	url=urllib.request.urlopen(Target)#urlopen返回 一个类文件对象
	page=url.read()#读取文件内容至pager
	url.close()
	page=page.decode('utf-8')#findall要求的对象格式为str
	page_text1=re.compile('<p class=""> (.+?)。')
	page_text2=re.compile('<p class=""> (.+?) ')

	for data in page_text1.findall(page):
		p=[]
		k=k+1
		d=mycut(data).strip()
		p=d.split(' ')
		for i in p:
			pp=re.compile('[\u4e00-\u9fa5]+')
			str=i
			if pp.findall(str):
				word.append(str)
			else:
				p.remove(str)
		
	for data in page_text2.findall(page):
		p=[]
		d=mycut(data).strip()
		p=d.split(' ')
		k=k+1
		for i in p:
			pp=re.compile('[\u4e00-\u9fa5]+')
			str=i
			if pp.findall(str):
				word.append(str)
			else:
				p.remove(str)	
	print(n)
####################################################################################
print(k)
print(len(word))
print(word)
for s in word:
	if s=='的':
		word.remove(s)
	elif s=='了':
		word.remove(s)
	elif s=='是':
		word.remove(s)
	elif s=='很':
		word.remove(s)
	elif s=='在':
		word.remove(s)
	elif s=='和':
		word.remove(s)
	elif s=='也':
		word.remove(s)
	elif s=='有':
		word.remove(s)
	elif s=='不':
		word.remove(s)
	elif s=='上':
		word.remove(s)
	elif s=='都':
		word.remove(s)
	elif s=='就':
		word.remove(s)
	
	


from collections import Counter
counts=Counter(word).items()
print(counts)


from pytagcloud import create_tag_image,make_tags
tags=make_tags(counts,maxsize=100)
create_tag_image(tags,'b.png',size=(1000,1000),fontname='MicrosoftYaHei',background=(0,0,0))
'''




for i in range(7):
########################################################################################
	n=n+1
	Target=web[i]
	url=urllib.request.urlopen(Target)#urlopen返回 一个类文件对象
	page=url.read()#读取文件内容至pager
	url.close()
	page=page.decode('utf-8')#findall要求的对象格式为str
	page_text1=re.compile('<p class=""> (.+?)。')
	page_text2=re.compile('<p class=""> (.+?) ')

	for data in page_text1.findall(page):
		w=[]
		p=[]
		k=k+1
		d=mycut(data).strip()
		p=d.split(' ')
		for i in p:
			pp=re.compile('[\u4e00-\u9fa5]+')
			str=i
			if pp.findall(str):
				w.append(str)
			else:
				p.remove(str)
			for s in w:
				if s=='的':
					w.remove(s)
				elif s=='了':
					w.remove(s)
				elif s=='是':
					w.remove(s)
				elif s=='很':
					w.remove(s)
		word.append(w)
	for data in page_text2.findall(page):
		w=[]
		p=[]
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
			for s in w:
				if s=='的':
					w.remove(s)
				elif s=='了':
					w.remove(s)
				elif s=='是':
					w.remove(s)
				elif s=='很':
					w.remove(s)
		word.append(w)
	print(n)
####################################################################################
print(k)
print(len(word))
print(word)

'''
from collections import Counter
counts=Counter(word).items()
print(counts)


from pytagcloud import create_tag_image,make_tags
tags=make_tags(counts,maxsize=100)
create_tag_image(tags,'b.png',size=(1000,1000),fontname='MicrosoftYaHei',background=(173,216,230))
'''

import gensim,logging 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#sentences =[['first','dog','fast'],['second','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog']] 
sentences=word
model = gensim.models.Word2Vec(sentences, min_count=1,size=100)  # 训练skip-gram模型; 默认window=5
###########################use the model##########################################################
print(model.similarity(u'电影',u'范冰冰'))# 计算两个词的相似度/相关程度
#print(model.most_similar(u'好'))# 计算某个词的相关词列表

y=model.most_similar(u'冯小刚',topn=10)
print('most similarity words with 冯小刚')
for item in y:
	print(item[0],'---->',item[1])
	print('\n')

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
# 保存模型，以便重用
model.save(u"书评.model")
# 对应的加载方式
# model_2 = word2vec.Word2Vec.load("text8.model")
'''


'''
film_dict = corpora.Dictionary(word)
film_corpus = [film_dict.doc2bow(i) for i in word]
film_lda = models.LdaModel(film_corpus, num_topics = 3, id2word = film_dict)
for i in range(3):
  film_lda.print_topic(i) #输出每个主题
'''
