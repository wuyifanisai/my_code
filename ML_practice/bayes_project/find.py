# -*- coding: utf-8 -*-'
import re
import pytagcloud
print('##########################################################################')
arr=[]
for i in range(5000):
	i=i+1
	txtpath=r'E:\%d.txt'%(i)
	fp=open(txtpath,'r',encoding= 'utf-8')
	for lines in fp.readlines():
		lines=lines.replace('\n','')
		lines=lines.split('  ')
		line=lines[0].split(' ')
		l=[]
		for i in line:
			pp=re.compile('[\u4e00-\u9fa5]+')
			str=i
			if pp.match(str):
				if len(str)>1:
					l.append(str)
				else:
					pass
			else:
				pass
		if l==[]:
			pass
		else:
			arr.append(l)
	fp.close()
print(len(arr))





import gensim 
from gensim import corpora,models
from gensim.models import word2vec
import logging 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#sentences =[['first','dog','fast'],['second','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog'],['first','dog']] 
sentences=arr
#sentences = word2vec.Text8Corpus(u"E:\word.txt",encoding= 'utf-8')
model = gensim.models.Word2Vec(sentences, size=200)  # 训练skip-gram模型; 默认window=5
###########################use the model##########################################################
#print(model.similarity(u'月',u'思'))# 计算两个词的相似度/相关程度
#print(model.most_similar(u'美女'))# 计算某个词的相关词列表


y=model.most_similar(u'美国',topn=10)
for item in y:
	print(item[0],item[1])

print('##############################')


y=model.most_similar(u'中国',topn=10)
for item in y:
	print(item[0],item[1])

print('##############################')

# 寻找对应关系
y3 = model.most_similar([u'中国'], [u'科技'], topn=5)
for item in y3:
    print(item[0], item[1])
print ("--------\n")


'''
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


count=[]
for i in range(100):
	i=i+1
	txtpath=r'E:\%d.txt'%(i)
	fp=open(txtpath,'r',encoding= 'utf-8')
	for lines in fp.readlines():
		lines=lines.replace('\n','')
		lines=lines.split('  ')
		line=lines[0].split(' ')
		for i in line:
			pp=re.compile('[\u4e00-\u9fa5]+')
			str=i
			if pp.match(str):
				if len(str)>1:
					count.append(str)
				else :
					pass
			else:
				pass
	fp.close()

from collections import Counter
counts=Counter(count).items()
#print(counts)
'''
from pytagcloud import create_tag_image,make_tags
tags=make_tags(counts,maxsize=100)
create_tag_image(tags,'word.png',size=(1000,1000),fontname='MicrosoftYaHei',background=(0, 0, 0))
'''

'''
film_dict = corpora.Dictionary(arr)
film_corpus = [film_dict.doc2bow(i) for i in arr]
film_lda = models.LdaModel(film_corpus, num_topics = 3, id2word = film_dict)
for i in range(3):
  print(film_lda.print_topic(i)) #输出每个主题
'''