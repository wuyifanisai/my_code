#coding: utf-8
from bs4 import BeautifulSoup
import urllib.request
import re

""" ***************使用bs提取网页中的标题、链接、图片***************"""
list_name=[]
list_score=[]
list2=[]
list3=[]
list4=[]
list5=[]
list6=[]

#read the html file from the web
url=urllib.request.urlopen('https://movie.douban.com/nowplaying/shanghai/')#urlopen返回 一个类文件对象
page=url.read()#读取文件内容至pager
url.close()
fp=open("grab.txt","wb")
fp.write(page)
fp.close()#将抓取的网页内容存至文件grab.txt文件中，以备不时之需

page=page.decode('utf-8')#findall要求的对象格式为str
print('************************************************')
print()
#解析网页
soup = BeautifulSoup(page)
for tag in soup.find_all(name="li", attrs={"class": re.compile("stitle")}):
	tal = tag.find(name="a", attrs={"target": re.compile("_blank")})
	pp=re.compile('[\u4e00-\u9fa5]+')
	#print('电影名称:',pp.findall(tal.string)[0])
	list_name.append(pp.findall(tal.string)[0])	

for tag in soup.find_all(name="li", attrs={"class": re.compile("srating")}):
	tal = tag.find(name="span", attrs={"class": re.compile("subject-rate")})
	if tal==None:
		#print('no score')
		list_score.append('no score')			
	else:
		#print(tal.string)
		list_score.append(tal.string)

for i in range(len(list_name )):
	if i<(len(list_score)):
		print('正在上映:',list_name[i],'豆瓣评分:',list_score[i])
		print()
		print('******************************************')
	else:
		print('即将上映:',list_name[i])
		print()
		print('******************************************')

