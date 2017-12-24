#coding: utf-8
import urllib.request
import re
""" ***************使用正则表达式提取网页中的标题、链接、图片***************"""
Target='https://movie.douban.com/'
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
page_title=re.compile('<meta name="keywords" content="(.+?)"/>')
s=s+" "+page_title.findall(page)[0]+"\n"
print(s)
page_name=re.compile('data-title="(.+?)" data-relea')
page_goal=re.compile('data-rate="(.+?)" data-star')
page_location=re.compile('data-region="(.+?)" data-director')

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
for data in page_name.findall(page):
	s=s+" "+data+"\n"
s=s+"\n"

for data in page_goal.findall(page):
	s=s+" "+data+"\n"


for i in range(len(page_goal.findall(page))):
	print(page_name.findall(page)[i])
	print(page_goal.findall(page)[i])
	print(page_location.findall(page)[i])
	print()






