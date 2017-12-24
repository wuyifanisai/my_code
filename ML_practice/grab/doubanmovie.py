#coding: utf-8
import urllib.request
import re
import pandas as pd
from sqlalchemy import create_engine
import pymysql
""" ***************使用正则表达式提取网页中的标题、链接、图片***************"""

web=['https://movie.douban.com/top250?start=0&filter=','https://movie.douban.com/top250?start=25&filter=','https://movie.douban.com/top250?start=50&filter=','https://movie.douban.com/top250?start=75&filter=','https://movie.douban.com/top250?start=100&filter=','https://movie.douban.com/top250?start=125&filter=','https://movie.douban.com/top250?start=150&filter=','https://movie.douban.com/top250?start=175&filter=','https://movie.douban.com/top250?start=200&filter=','https://movie.douban.com/top250?start=225&filter=']
print(web)
for i in range(len(web)):
	Target=web[i]
	url=urllib.request.urlopen(Target)#urlopen返回 一个类文件对象
	page=url.read()#读取文件内容至pager
	url.close()

	page=page.decode('utf-8')#findall要求的对象格式为str
#page=page.decode('gbk')#findall要求的对象格式为str

	s=""
	s=s+"标题：\n"
	page_name=re.compile('<span class="title">(.+?)</span>')
	page_location=re.compile('nbsp;/&nbsp;(.+?)&nbsp;/&nbsp')
	page_id=re.compile('<em class="">(.+?)</em>')
	page_score=re.compile('average">(.+?)</span>')
	page_year=re.findall(r'\d{4}&',page)
	page_type=re.compile(u"[\u4e00-\u9fa5]{2}&nbsp;/&nbsp;([\u4e00-\u9fa5]{2}.*)")

	yy=[]

	n=0
	for y in page_year:
		y=y[0:4]
		if int(y)>2017:
			pass
		elif int(y)<1800:
			pass
		else:
			yy.append(int(y))
			n=n+1
	print(n)
	if i==3:
		yy.insert(8,1961)
	print(yy)
	name=[]
	k=0
	for data in page_name.findall(page):
		if 'nbsp' in data:
			pass
		else :
			print(data)
			name.append(data)
			k=k+1

	print(k)
	data_name=pd.Series(name)
	data_location=pd.Series(page_location.findall(page))
	score=[]
	for i in range(25):
		score.append((page_score.findall(page)[i]))

	k=0
	type=[]
	for data in page_type.findall(page):
		print(data)
		type.append(data)
		k=k+1
	print(k)
	'''
	data={'movie':name,'type':type,'location':page_location.findall(page),'year':yy,'score':score}
	d=pd.DataFrame(data)
	print(d)
	d.to_excel('E:\datamovie02.xls')
	print(name)
	'''
####################################################################################################

	conn=pymysql.connect(host='192.168.1.104',port=3306,user='root',passwd='123456',db='data1205',charset='utf8')
	cur=conn.cursor()
	cur.execute("create table if not exists doubandata(location varchar(20),movie varchar(20),score float,type varchar(50),year int)")
	for i in range(len(name)):
		cur.execute("insert into doubandata(location,movie,score,type,year) values(%s,%s,%s,%s,%s)",(page_location.findall(page)[i],name[i],score[i],type[i],yy[i]))

	cur.close()
	conn.commit()
	conn.close()
