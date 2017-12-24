# -*- coding: utf-8 -*-'
def classfy_bayes(SENTENSE,class_of_the_sentense):
	import re
	import pytagcloud
	import jieba #导入结巴分词，需要自行下载安装
	import random
	import numpy as np
	import pandas as pd
	import math

	print('##########################################################################')
	k1=0
	txtpath=r'E:\Master\PPDAMcode\bayes_project\%d.txt'%(1)
	fp=open(txtpath,'r',encoding= 'gbk')

	i=0
	words_tiyu=[]

	for lines in fp.readlines():
		w=[]
		words1=[]
		i=i+1
		lines=lines.replace('\n','')
		lines=lines.split(' ')
		if i<1000:
			for word in lines:
				if re.match(u"[\u4e00-\u9fa5]{1,3}",word):
					if len(word)<5:
						w.append(word)
					else:
						for str in list(jieba.cut(word)):
							w.append(str)
	
		for str in w:
			if re.match(u"[\u4e00-\u9fa5]{1,3}",str):
				words1.append(str)
		if len(words1)>1:
			words_tiyu.append(words1)
			k1=k1+1

	print('##########################################################################')
	k2=0
	txtpath=r'E:\Master\PPDAMcode\bayes_project\%d.txt'%(2)
	fp=open(txtpath,'r',encoding= 'gbk')

	i=0
	words_yule=[]

	for lines in fp.readlines():
		w=[]
		words2=[]
		i=i+1
		lines=lines.replace('\n','')
		lines=lines.split(' ')
		if i<1000:
			for word in lines:
				if re.match(u"[\u4e00-\u9fa5]{1,3}",word):
					if len(word)<5:
						w.append(word)
					else:
						for str in list(jieba.cut(word)):
							w.append(str)
	
		for str in w:
			if re.match(u"[\u4e00-\u9fa5]{1,3}",str):
				words2.append(str)
		if len(words2)>1:
			words_yule.append(words2)
			k2=k2+1

	print('##########################################################################')
	k3=0
	txtpath=r'E:\Master\PPDAMcode\bayes_project\%d.txt'%(3)
	fp=open(txtpath,'r',encoding= 'gbk')

	i=0
	words_policy=[]

	for lines in fp.readlines():
		w=[]
		words3=[]
		i=i+1
		lines=lines.replace('\n','')
		lines=lines.split(' ')
		if i<1000:
			for word in lines:
				if re.match(u"[\u4e00-\u9fa5]{1,3}",word):
					if len(word)<7:
						w.append(word)
					else:
						for str in list(jieba.cut(word)):
							w.append(str)
	
		for str in w:
			if re.match(u"[\u4e00-\u9fa5]{1,3}",str):
				words3.append(str)
		if len(words3)>1:
			words_policy.append(words3)
			k3=k3+1


#############################################################################
#将上述三个词条集合整合成一个总得集合，并去掉重复的词条

	sentense=[]
	wordlist=[]

	for str in words_tiyu:
		if str not in sentense:
			sentense.append(str)
		else:
			pass

	for str in words_yule:
		if str not in sentense:
			sentense.append(str)
		else:
			pass

	for str in words_policy:
		if str not in sentense:
			sentense.append(str)
		else:
			pass

	for i in range(len(sentense)):
		for j in range(len(sentense[i])):
			if sentense[i][j] not in wordlist:
				wordlist.append(sentense[i][j])

	random.shuffle(wordlist)
	print('词库数量：',len(wordlist))
#print(wordlist)#wordlist is the vocablist
############################################################################
#构建进行向量化操作method
	def setofwordsvec1(wordlist,inputset):
		k=0
		vec=[0]*len(wordlist)
		for word in inputset:
			if word in wordlist:
				vec[wordlist.index(word)]=vec[wordlist.index(word)]+1
			else:
				k=k+1
				#print("the word :%s is not in my wordlist!"%word)
		print('这句话的',100*(1-k/len(inputset)),'%被用于分析计算')
		return vec

	def setofwordsvec(wordlist,inputset):
		k=0
		vec=[0]*len(wordlist)
		for word in inputset:
			if word in wordlist:
				vec[wordlist.index(word)]=vec[wordlist.index(word)]+1
			else:
				pass
		return vec
############################################################################
#生成train_matrix
	train_matrix=[]
	for i in range(len(words_tiyu)):
		train_matrix.append(setofwordsvec(wordlist,words_tiyu[i]))

	for i in range(len(words_yule)):
		train_matrix.append(setofwordsvec(wordlist,words_yule[i]))

	for i in range(len(words_policy)):
		train_matrix.append(setofwordsvec(wordlist,words_policy[i]))


############################################################################
#生成样本标签向量
	train_category=[]
	category_1=[1]*k1   #k1 类别1的样本数
	category_2=[2]*k2	  #k2 类别2的样本数
	category_3=[3]*k3   #k3 类别3的样本数

	for i in range(len(category_1)):
		train_category.append(category_1[i])

	for i in range(len(category_2)):
		train_category.append(category_2[i])

	for i in range(len(category_3)):
		train_category.append(category_3[i])

###################################################################################
#朴素贝叶斯分类器训练方法代码,

	def trainbayes(train_matrix,train_category):
		num_train=len(train_matrix)
		num_word=len(train_matrix[0])
		p_tiyu=len(words_tiyu)/len(train_matrix)
		p_yule=len(words_yule)/len(train_matrix)
		p_policy=len(words_policy)/len(train_matrix)

		p_1_num=np.zeros(num_word)#当类别为1时，词条向量上各个词条出现的概率
		p_2_num=np.zeros(num_word)#当类别为2时，词条向量上各个词条出现的概率
		p_3_num=np.zeros(num_word)#当类别为3时，词条向量上各个词条出现的概率

		num_1=2
		num_2=2
		num_3=2

		for i in range(num_train):
			if train_category[i]==1:
				p_1_num=p_1_num+train_matrix[i]
				num_1=num_1+1

			elif train_category[i]==2:
				p_2_num=p_2_num+train_matrix[i]
				num_2=num_2+1

			else:
				p_3_num=p_3_num+train_matrix[i]
				num_3=num_3+1

		p_1vect=p_1_num/num_1
		p_2vect=p_2_num/num_2
		p_3vect=p_3_num/num_3

		return p_tiyu,p_yule,p_policy,p_1vect,p_2vect,p_3vect

#########################################################################
#根据上述生成的train_matrix,train_category， setofwordsvec函数，trainbayes函数，构建最后的分类机器函数
#1---tiyu,2---yule,3---policy
	def classfyNB(vec_to_class,p_tiyu,p_yule,p_policy,p_1vect,p_2vect,p_3vect):
		p_is_tiyu=sum(vec_to_class*p_1vect)*(1000*p_tiyu)

		p_is_yule=sum(vec_to_class*p_2vect)*(1000*p_yule)

		p_is_policy=sum(vec_to_class*p_3vect)*(1000*p_policy)

		if p_is_tiyu>p_is_yule:
			if p_is_tiyu>p_is_policy:
				print('这句话是关于体育方面!')
			else:
				print('这句话是关于政治军事方面!')
		else:
			if p_is_yule>p_is_policy:
				print('这句话是关于娱乐圈方面!')
			else:
				print('这句话是关于政治军事方面!')
		print('体育',p_is_tiyu/(p_is_tiyu+p_is_yule+p_is_policy),' 娱乐 ',p_is_yule/(p_is_tiyu+p_is_yule+p_is_policy),'  政治 ',p_is_policy/(p_is_tiyu+p_is_yule+p_is_policy))
		return p_is_tiyu/(p_is_tiyu+p_is_yule+p_is_policy),p_is_yule/(p_is_tiyu+p_is_yule+p_is_policy),p_is_policy/(p_is_tiyu+p_is_yule+p_is_policy)
#########################################################################
#构造最终的样本测试主函数

	def testing_main(input_sentense,train_matrix,train_category,wordlist):
		p_tiyu,p_yule,p_policy,p_1vect,p_2vect,p_3vect=trainbayes(train_matrix,train_category)#获得相关参数
		input_sentense=list(jieba.cut(input_sentense))#将输入的句子信息转化为词条列表
		input_sentense=setofwordsvec1(wordlist,input_sentense)#将词条列表转化为向量
		p_1,p_2,p_3=classfyNB(input_sentense,p_tiyu,p_yule,p_policy,p_1vect,p_2vect,p_3vect)#进行分类计
		return p_1,p_2,p_3

#########################测试#############################################
	s=testing_main(SENTENSE,train_matrix,train_category,wordlist)


#########################学习程序#############################################

	if class_of_the_sentense >0:
		if s[class_of_the_sentense-1]==max(s):
			print('#####################################################+')
			print('###################以下为初次识别正确的自学习过程######################+')
			if max(s)<0.6:
				f=open(r'E:\Master\PPDAMcode\bayes_project\%d.txt'%(class_of_the_sentense),'a') #输入的数字代表人为判断的类别
				f.write('\n')
				ss=list(jieba.cut(SENTENSE))
				for str in ss:
					p=testing_main(str,train_matrix,train_category,wordlist)[class_of_the_sentense-1]
					if p>0.6:
						f.write(str)
						print()
						print('**********************自动学习了词条：',str)
						print()
				f.close()

###############以下为初次识别错误从而进行反复学习的过程###############################
		learn=0
		while s[class_of_the_sentense-1]!=max(s)and(learn != 4):
			learn=learn+1
			print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
			print('分析错误，进行反省学习。。。。。。。。。。。。。')
			print()
			print()
			print()
#############需要设定一个自主选择性学习的功能，只针对关键的，有效的词条进行吸收学习
			f=open(r'E:\Master\PPDAMcode\bayes_project\%d.txt'%(class_of_the_sentense),'a') #输入的数字代表人为判断的类别
			f.write('\n')

			sss=list(jieba.cut(SENTENSE))
			count_learn=0
			for str in sss:
				p=testing_main(str,train_matrix,train_category,wordlist)[class_of_the_sentense-1]
				if p>0.6:
					f.write(str)
					print()
					print('*******************************************************自动学习了词条：',str)
					print()
					count_learn=count_learn+1
			if count_learn==0:
		 #输入的数字代表人为判断的类别
				f.write(SENTENSE)
			f.close()
	################################################################################
	#################################################################################
	
			print('##########################################################################')
			k1=0
			txtpath=r'E:\Master\PPDAMcode\bayes_project\%d.txt'%(1)
			f=open(txtpath,'r',encoding= 'gbk')

			i=0
			words_tiyu=[]

			for lines in f.readlines():
				w=[]
				words1=[]
				i=i+1
				lines=lines.replace('\n','')
				lines=lines.split(' ')
				if i<1000:
					for word in lines:
						if re.match(u"[\u4e00-\u9fa5]{1,3}",word):
							if len(word)<5:
								w.append(word)
							else:
								for str in list(jieba.cut(word)):
									w.append(str)
	
				for str in w:
					if re.match(u"[\u4e00-\u9fa5]{1,3}",str):
						words1.append(str)
				if len(words1)>1:
					words_tiyu.append(words1)
					k1=k1+1
			f.close()
			print('##########################################################################')
			k2=0
			txtpath=r'E:\Master\PPDAMcode\bayes_project\%d.txt'%(2)
			f=open(txtpath,'r',encoding= 'gbk')

			i=0
			words_yule=[]

			for lines in f.readlines():
				w=[]
				words2=[]
				i=i+1
				lines=lines.replace('\n','')
				lines=lines.split(' ')
				if i<1000:
					for word in lines:
						if re.match(u"[\u4e00-\u9fa5]{1,3}",word):
							if len(word)<5:
								w.append(word)
							else:
								for str in list(jieba.cut(word)):
									w.append(str)
	
				for str in w:
					if re.match(u"[\u4e00-\u9fa5]{1,3}",str):
						words2.append(str)
				if len(words2)>1:
					words_yule.append(words2)
					k2=k2+1
			f.close()
			print('##########################################################################')
			k3=0
			txtpath=r'E:\Master\PPDAMcode\bayes_project\%d.txt'%(3)
			f=open(txtpath,'r',encoding= 'gbk')
			i=0
			words_policy=[]

			for lines in f.readlines():
				w=[]
				words3=[]
				i=i+1
				lines=lines.replace('\n','')
				lines=lines.split(' ')
				if i<1000:
					for word in lines:
						if re.match(u"[\u4e00-\u9fa5]{1,3}",word):
							if len(word)<7:
								w.append(word)
							else:
								for str in list(jieba.cut(word)):
									w.append(str)
	
				for str in w:
					if re.match(u"[\u4e00-\u9fa5]{1,3}",str):
						words3.append(str)
				if len(words3)>1:
					words_policy.append(words3)
					k3=k3+1
			f.close()

#############################################################################
#将上述三个词条集合整合成一个总得集合，并去掉重复的词条

			sentense=[]
			wordlist=[]

			for str in words_tiyu:
				if str not in sentense:
					sentense.append(str)
				else:
					pass
	
			for str in words_yule:
				if str not in sentense:
					sentense.append(str)
				else:
					pass

			for str in words_policy:
				if str not in sentense:
					sentense.append(str)
				else:
					pass
	
			for i in range(len(sentense)):
				for j in range(len(sentense[i])):
					if sentense[i][j] not in wordlist:
						wordlist.append(sentense[i][j])

			random.shuffle(wordlist)
			print('词库数量：',len(wordlist))
	#print(wordlist)#wordlist is the vocablist


############################################################################
#生成train_matrix
			train_matrix=[]
			for i in range(len(words_tiyu)):
				train_matrix.append(setofwordsvec(wordlist,words_tiyu[i]))

			for i in range(len(words_yule)):
				train_matrix.append(setofwordsvec(wordlist,words_yule[i]))

			for i in range(len(words_policy)):
				train_matrix.append(setofwordsvec(wordlist,words_policy[i]))


############################################################################
#生成样本标签向量
			train_category=[]
			category_1=[1]*k1   #k1 类别1的样本数
			category_2=[2]*k2	  #k2 类别2的样本数
			category_3=[3]*k3   #k3 类别3的样本数

			for i in range(len(category_1)):
				train_category.append(category_1[i])

			for i in range(len(category_2)):
				train_category.append(category_2[i])

			for i in range(len(category_3)):
				train_category.append(category_3[i])


			s=testing_main(SENTENSE,train_matrix,train_category,wordlist)
	##################################################################################
	##################################################################################

