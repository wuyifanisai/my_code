#coding: utf-8
import urllib.request
import re
""" ***************使用正则表达式提取网页中的标题、链接、图片***************"""
Target='http://goal.sports.163.com/54/stat/standings/2016_3.html'
url=urllib.request.urlopen(Target)#urlopen返回 一个类文件对象
page=url.read()#读取文件内容至pager

url.close()
fp=open("grab.txt","wb")
fp.write(page)
fp.close()#将抓取的网页内容存至文件grab.txt文件中，以备不时之需
page=page.decode('utf-8')#findall要求的对象格式为str
#page=page.decode('gbk')#findall要求的对象格式为str

s=""
s=s+"标题：\n"
page_title=re.compile('<title>(.+?)</title>')
s=s+" "+page_title.findall(page)[0]+"\n"#提取标题
s=s+"图片：\n"
page_image=re.compile('<img src=\"(.+?)\"')
page_num=re.compile('<th>(.+?)</th>')
page_data=re.compile('<td>(.+?)</td>')
page_team=re.compile('">(.+?)</a></td>') 
page_team=re.compile('">(.+?)</a></td>') 
'''
for data in page_image.findall(page):
	s=s+" "+data+"\n"#提取图片
s=s+"链接：\n"
'''
page_link=re.compile('href=\"(.+?)\"')
'''
for data in page_link.findall(page):
	if "http" in data:	
		s=s+" "+data+"\n"#提取链接
'''
for data in page_num.findall(page):
	s=s+" "+data+"\n"
s=s+"\n"

for data in page_team.findall(page):
	s=s+" "+data+"\n"
print(s)
n=0
for data in page_data.findall(page):
	if (len(data))<5:
		s=s+" "+data+" \n"
		print(float(data))
		n=n+1
		if n>8:
			if((n)%9==0):
				print('\n')



