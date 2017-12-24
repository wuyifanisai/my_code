#coding: utf-8
from bs4 import BeautifulSoup
import urllib.request
import re

""" ***************使用bs提取网页中的标题、链接、图片***************"""
list0=[]
list1=[]
list2=[]
list3=[]
list4=[]
list5=[]
list6=[]

                                
for i in range(1,6):
	#read the html file from the web
	url=urllib.request.urlopen('http://cs.fang.lianjia.com/loupan/pg'+str(i))#urlopen返回 一个类文件对象
	page=url.read()#读取文件内容至pager
	url.close()
	fp=open("grab.txt","wb")
	fp.write(page)
	fp.close()#将抓取的网页内容存至文件grab.txt文件中，以备不时之需

	page=page.decode('utf-8')#findall要求的对象格式为str

	#解析网页
	soup = BeautifulSoup(page)

	#提取楼盘的名字
	for tag in soup.find_all(name="div", attrs={"class": re.compile("col-1")}):
		ta1 = tag.find(name="a", attrs={"target": re.compile("_blank")})
		list0.append('长沙')
		list1.append(ta1.string)
	#提取建筑面积字段
		ta2 = tag.find(name="div", attrs={"class": re.compile("area")})
		t2 = ta2.find(name="span")
		if t2 != None:
			list2.append(t2.string[3:])
		else:
			list2.append(0)
	#提取在售状态字段
		ta3 = tag.find(name="span", attrs={"class": re.compile("onsold")})
		list3.append(ta3.string)
	#提取住宅类型字段
		ta4 = tag.find(name="span", attrs={"class": re.compile("live")})
		list4.append(ta4.string)

	#another tag----------------------------------------------------------------------------------------------
	#提取每平米均价字段
	for tag in soup.find_all(name="div", attrs={"class": re.compile("col-2")}):
		ta5 = tag.find(name="span", attrs={"class": re.compile("num")})
		if ta5 != None:
			list5.append(ta5.string)
		else:
			list5.append(0)
	#提取总价字段
		ta6 = tag.find(name="div", attrs={"class": re.compile("sum-num")})
		if ta6 !=None:
			t6 = ta6.find(name="span")
			list6.append(t6.string)
		else:
			list6.append(0)
print(list6)

