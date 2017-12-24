  #coding: utf-8

import urllib.request
import re
""" ***************使用正则表达式提取网页中的标题、链接、图片***************"""
Target='http://sports.baidu.com/'
url=urllib.request.urlopen(Target)#urlopen返回 一个类文件对象
page=url.read()#读取文件内容至pager

url.close()
fp=open("grab.txt","wb")
fp.write(page)
fp.close()#将抓取的网页内容存至文件grab.txt文件中，以备不时之需

page=page.decode('gbk')#findall要求的对象格式为str

s=""
s=s+"标题：\n"
page_title=re.compile('<title>(.+?)</title>')
s=s+" "+page_title.findall(page)[0]+"\n"#提取标题
s=s+"图片：\n"
page_image=re.compile('<img src=\"(.+?)\"')
page_link=re.compile('href=\"(.+?)\"')
page_news=re.compile('_blank">(.+?)</a></li>')

n=0
for data in page_news.findall(page):
	n=n+1
	s=s+" "+data+"\n"
	s=s+"\n"
	if n>100:
		break
print(s)





