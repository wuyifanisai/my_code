from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

#txtpath='E:\\Master\\PPDAMcode\\page_recommended_PROJECT\\mid_page\\mid_page_theme_wordlist.txt'
txtpath='E:\\3.txt'
fp=open(txtpath,'r',encoding= 'gbk')
sentense=[]

for line in fp.readlines():
	sentense.append(line)
#print(sentense)
#sentense=["我 来到 你的 城市","你 爱 数据 挖掘","它 爱 吃 红烧肉"]

c=CountVectorizer()
t=TfidfTransformer()

vec=c.fit_transform(sentense)# 接收sentense中所有出现的词汇，进行停用词汇过滤后形成词袋
#print(vec)

tfidf=t.fit_transform(vec) #将词袋中的词汇计算权重
weight=tfidf.toarray()
#print(weight)

word=c.get_feature_names()
#print(word)

new_wordbag=[]
allthemeword=[]

for i in range(len(weight)):
	w=[]
	print(i,'#########################################################################')
	for j in range(len(weight[i])):
		if weight[i][j]>0.2:#通过权重阈值的筛选，将每一个文档特征词中权重过小的丢掉
			print(word[j],'---->',weight[i][j])
			w.append(word[j])
			if word[j] not in new_wordbag:
				new_wordbag.append(word[j])

	allthemeword.append(w)
	

print('每个文档包含的词汇........',allthemeword)
print(word)
print(len(new_wordbag),len(word))
