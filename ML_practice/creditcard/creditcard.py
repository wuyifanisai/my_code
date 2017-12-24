import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
import time,datetime

data = pd.read_csv('E:\\Master\\PPDAMcode\\PROJECT\\creditcard\\creditcard.csv')
#print(data.columns)
#print(data.Class.value_counts())

'''
for s in list(data.columns):
    plt.scatter(data[s],data.Class)
    plt.title(s)
    plt.show()
'''

df=data.drop('Class',axis=1)
label = data['Class']


df = (df-df.min())/(df.max() - df.min())


###################################### using resampling and weight adjust to solve unbalance data #############################
############################################################################################################################

def get_the_data(df,label,train_size0,train_size1,undersampling0,oversampling1):
    # 为了保证测试集合中两个类别的比例与原始数据中的比例一致，保证测试集合的真实还原性
    df_1=df[label==1]
    label_1=label[label==1]

    df_0=df[label==0]
    label_0=label[label==0]


    # 分别从两个类别中抽取出测试集合与训练集合
    import random

    index_train=int(train_size0*len(df_0))

    df_0_train=df_0.iloc[:index_train,:]
    label_0_train=label_0[:index_train]

    df_0_test=df_0.iloc[index_train:,:]
    label_0_test=label_0[index_train:]


    index_train=int(train_size1*len(df_1))

    df_1_train=df_1.iloc[:index_train,:]
    label_1_train=label_1[:index_train]

    df_1_test=df_1.iloc[index_train:,:]
    label_1_test=label_1[index_train:]


    df_0_train.index=range(len(df_0_train))
    label_0_train.index=range(len(label_0_train))
    df_1_train.index=range(len(df_1_train))
    label_1_train.index=range(len(label_1_train))

    df_0_test.index=range(len(df_0_test))
    label_0_test.index=range(len(label_0_test))
    df_1_test.index=range(len(df_1_test))
    label_1_test.index=range(len(label_1_test))



    #然后对两个类别中的训练集合进行重采样处理----------------------------------

    #对多数量的类别进行undersampling
    index_undersampling=random.sample(list(df_0_train.index),int(undersampling0*len(df_0_train)))
    df_0_train=df_0_train.iloc[index_undersampling,:]
    label_0_train=label_0_train.loc[index_undersampling]


    #对少数量的类别进行oversampling
    df_1_train_=df_1_train.copy()
    label_1_train_=label_1_train.copy()

    if oversampling1 >=1:
        for i in range(oversampling1):
            df_1_train_=pd.concat((df_1_train_,df_1_train),axis=0)
            label_1_train_=pd.concat((label_1_train_,label_1_train),axis=0)


    print('data in train:')
    print('sample of 0 num:',len(df_0_train))
    print('sample of 1 num:',len(df_1_train_))

    df_train = pd.concat((df_0_train,df_1_train_),axis=0)
    label_train = pd.concat((label_0_train,label_1_train_),axis=0)


    df_test = pd.concat((df_0_test,df_1_test),axis=0)
    label_test = pd.concat((label_0_test,label_1_test),axis=0)

    print('df_train.shape:',df_train.shape)
    print('label_train.shape:',label_train.shape)
    print('df_test.shape:',df_test.shape)
    print('label_test.shape:',label_test.shape)

    df_train.index=range(len(df_train))
    label_train.index=range(len(label_train))
    df_test.index=range(len(df_test))
    label_test.index=range(len(label_test))

    return df_train,label_train,df_test,label_test




def evaluate(label_test,y_pred):

    TP=0
    TN=0
    FP=0
    FN=0

    for i in range(len(label_test)):
        if label_test[i]==1:
            if y_pred[i] == 1:
                TP=TP+1
            else:
                FN=FN+1
        else:
            if y_pred[i] ==0:
                TN=TN+1
            else:
                FP=FP+1
    
    print('TP:',TP,'FN:',FN)
    print('FP:',FP,'TN:',TN)

    print('sample of 1-->','precision:',TP/(TP+FP),'recall',TP/(TP+FN))
    print('sample of 0-->','precision:',TN/(TN+FN),'recall',TN/(TN+FP))
    
    from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
    #print('roc:',roc_auc_score(label_test,y_pred))
    print('precision:',precision_score(label_test,y_pred))
    print('recall:',recall_score(label_test,y_pred))

    return precision_score(label_test,y_pred),recall_score(label_test,y_pred),f1_score(label_test,y_pred)

'''
######################################## do some feature selection #################################################
df_train,label_train,df_test,label_test=get_the_data(df,label,0.6,0.6,0.01,1)
'''

######################################### modeling ##############################################
'''
#xgb -----------------------
df_train,label_train,df_test,label_test=get_the_data(df,label,0.5,0.5,0.01,1)
from xgboost.sklearn import XGBRegressor
model=XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=50,
                   silent=True, objective="binary:logistic",
                   subsample=0.9,scale_pos_weight=3)

model.fit(df_train,label_train)
y_pred=list(model.predict(df_test))

# set the threshold
for i in range(len(y_pred)):
    if y_pred[i] > 0.65:
        y_pred[i]=1
    else:
        y_pred[i]=0

evaluate(label_test,y_pred)


# rf----------------------
print('RF----------')
df_train,label_train,df_test,label_test=get_the_data(df,label,0.6,0.6,0.01,1)
model1 = RandomForestClassifier(n_estimators=50, max_depth=3, max_features=0.9,class_weight={0:1,1:5})
model1.fit(df_train,label_train)
y_pred = list(model1.predict(df_test))
evaluate(label_test,y_pred)

#用RF建模对不平衡数据进行分类，通过类别权重调整与重采样处理，发现少类别的召回率达到80%以上，但是查准率很低

print('************************************')
'''


'''
# blagging ----------------------------
df_train,label_train,df_test,label_test=get_the_data(df,label,0.1,0.1,0.05,10)

from blagging_ import BlaggingClassifier
model = BlaggingClassifier( 
                            #base_estimator = LogisticRegression(C=1,max_iter=100,penalty='l2',class_weight={0:1,1:1},random_state=10),
                            base_estimator = RandomForestClassifier(n_estimators=100, max_depth=3, class_weight={0:1,1:7}),
                            #base_estimator = GradientBoostingClassifier(n_estimators=100,max_depth=3,),
                            #base_estimator = XGBClassifier(n_estimators=100,max_depth=3),

                            n_estimators = 500 #500
                           )
model.fit(df_train,label_train)
y_pred = list(model.predict(df_test))
evaluate(label_test,y_pred)
C=1/0

p=[]
r=[]
f=[]

#for n in list(np.arange(0.2,0.6,0.01)):
for n in list(np.arange(0.4,0.6,0.01)):
    print('**********',n,'***********')
    y_pred = list(model.predict_proba(df_test))
    for i in range(len(y_pred)):
        if y_pred[i][1] > n:
            y_pred[i]=1
        else:
            y_pred[i]=0
    try:
        a,b,c=evaluate(label_test,y_pred)
        p.append(a)
        r.append(b)
        f.append(c)
    except:
        print('error !')

pd.Series(p).plot()
pd.Series(r).plot()
pd.Series(f).plot()
plt.scatter(p,r,marker='*')
plt.show()

c=1/0
'''


'''
# lr-------------------------------
print('LR--------------')
df_train,label_train,df_test,label_test=get_the_data(df,label,0.5,0.5,0.4,4)

model2=LogisticRegression(C=1,max_iter=100,penalty='l2',class_weight={0:1,1:15},random_state=10)
model2.fit(df_train,label_train)
y_pred = list(model2.predict_proba(df_test))

print(y_pred)
# set the threshold
p=[]
r=[]
f=[]
for n in list(np.arange(0.8,0.9,0.01)):
    print('**********',n,'***********')
    y_pred = list(model2.predict_proba(df_test))

    try:
        for i in range(len(y_pred)):
            if y_pred[i][1] > n:
                y_pred[i]=1
            else:
                y_pred[i]=0
        a,b,c=evaluate(label_test,y_pred)
        p.append(a)
        r.append(b)
        f.append(c)

    except:
        print('some is 0 !')


print('**********************************')
pd.Series(p).plot()
pd.Series(r).plot()
pd.Series(f).plot()
plt.scatter(p,r,marker='*')
plt.show()

C=1/0
'''


'''
# SVM ------------------------------
df_train,label_train,df_test,label_test=get_the_data(df,label,0.5,0.5,0.01,3)
model = svm.SVC(C=0.1,kernel='rbf',degree=2,gamma=0.001,class_weight={0:1,1:1})
model.fit(df_train,label_train)

y_pred = model.predict(df_test)
evaluate(label_test,y_pred)

from sklearn.metrics import recall_score
print('recall:',recall_score(label_test,y_pred))

'''

'''
# DNN ---------------------------------
df_train,label_train,df_test,label_test=get_the_data(df,label,0.7,0.7,0.01,1)
df_train,label_train,df_test,label_test=df_train.as_matrix(),label_train.as_matrix(),df_test.as_matrix(),label_test.as_matrix()
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout

model = Sequential()

model.add(Dense(20, input_dim=30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
model.fit(df_train, label_train, nb_epoch = 100, batch_size = 20,class_weight={0:1,1:10})

y_pred=list(model.predict_classes(df_test).reshape(len(df_test)))
print(y_pred)
evaluate(label_test,y_pred)
'''

'''
# blagging_2 method ----------------------------------------------
df_train,label_train,df_test,label_test=get_the_data(df,label,0.1,0.1,0.5,1)
def blagging2(base_model,df,label,n_estimators):
    import random
    n=n_estimators
    model = base_model
    model.warm_start = True

    df_0 = df[label==0]
    df_1 = df[label==1]
    label_0 = label[label==0]
    label_1 = label[label==1]

    for i in range(n):
        index = random.sample(range(len(df_0)),len(df_1))

        df_0_ = df_0.iloc[index,:]
        label_0_ = label_0[index]

        df= pd.concat((df_0_,df_1),axis=0)
        label = pd.concat((label_0_,label_1),axis=0)

        print(len(df))
        print(len(label))
        model.fit(df,label)

        print('----------- num_train:',i,'---------------------')


    return model

model = blagging2(LogisticRegression(C=1,max_iter=100,penalty='l2',class_weight={0:1,1:1},random_state=10,),
                  #RandomForestClassifier(n_estimators=100, max_depth=3, class_weight={0:1,1:7}),

                  df_train,label_train,100)

y_pred = list(model.predict(df_test))

p=[]
r=[]
f=[]
for n in list(np.arange(0.2,0.7,0.01)):
    print('**********',n,'***********')
    y_pred = list(model.predict_proba(df_test))

    try:
        for i in range(len(y_pred)):
            if y_pred[i][1] > n:
                y_pred[i]=1
            else:
                y_pred[i]=0
        a,b,c=evaluate(label_test,y_pred)
        p.append(a)
        r.append(b)
        f.append(c)

    except:
        print('some is 0 !')


print('**********************************')
pd.Series(p).plot()
pd.Series(r).plot()
pd.Series(f).plot()
plt.scatter(p,r,marker='*')
plt.show()

c=1/0
'''


'''
通常可以将正负样本权重调整与欠采样同时应用在不平衡数据分类问题上，权重的数值可以根据正负样本的数量比例来确定，也可以通过欠采样的方式
将数量较多的样本数量降低
只通过权重调整可以改良不平衡数据的分类效果，但是在正负样本极其不平衡的情况下，权重调整无法将效果改良到特别好
只通过欠采样也可以有效提高分类效果
一般通过查准率以及查全率来评估不平衡数据分类的效果，一般可以着重看较少样本的召回率，这个值比较重要
可以将欠采样与权重调整结合起来改进不平衡数据的分类效果

过采样能够很好的提高不平衡数据的分类效果

将重采样，欠采样，权重调整三者结合起来用于处理数据不平衡分类问题，能够有不错的效果

'''


###################################### using discrete point analysis to solve unbalance data ###################################
############################################################################################################################v

df_train,label_train,df_test,label_test=get_the_data(df,label,0.04,0.04,1,0)

'''
from sklearn.manifold import TSNE
model = TSNE(n_components=2,random_state=1)
d=pd.DataFrame(model.fit_transform(df_train))

d0=d[label_train==0]
d1=d[label_train==1]

x0,y0=d0.iloc[:,0],d0.iloc[:,1]
x1,y1=d1.iloc[:,0],d1.iloc[:,1]

plt.scatter(x0,y0,marker='o',edgecolors='y',s=20)
plt.scatter(x1,y1,marker='*',edgecolors='b',s=100)
plt.show()

c=1/0
'''
'''
from sklearn.cluster import KMeans
k=5
model = KMeans(n_clusters=k, init='k-means++',n_init=10, max_iter=300,tol=0.0001,verbose=0,)
model.fit(df_train)


#简单打印结果
r1 = pd.Series(model.labels_).value_counts() #统计各个类别的数目
r2 = pd.DataFrame(model.cluster_centers_) #找出聚类中心

clusters = pd.concat([r2, r1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
clusters.columns = list(df_train.columns) + [u'类别数目'] #重命名表头

#详细输出原始数据及其类别
r = pd.concat([df_train, pd.Series(model.labels_, index = df_train.index)], axis = 1)  #详细输出每个样本对应的类别
r.columns = list(df_train.columns) + [u'聚类类别'] #重命名表头

#print(r)

for i in range(k):
	print('label of sample in cluster',i)
	cluster_index=list(r[r[u'聚类类别'] == i].index)
	print(label_train[cluster_index].value_counts())
'''
print('************************ DBSCAN  ********************************')
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.15,min_samples=20,)
model.fit(df_train)
y_hat = list(model.labels_)

print(pd.Series(y_hat).value_counts())
print()
y_hat=pd.DataFrame(y_hat,columns=['cluster'],index= range(len(y_hat)))


y_hat[y_hat.cluster > -1] =0
y_hat[y_hat.cluster == -1] =1


from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
#print('roc:',roc_auc_score(label_test,y_pred))
print('precision:',precision_score(label_train,y_hat['cluster']))
print('recall:',recall_score(label_train,y_hat['cluster']))


################### second ###############
print('second cluster ----------------')
df_train_= df_train[y_hat.cluster == 1]
label_train_ =label_train[y_hat.cluster == 1]

df_train_.index = range(len(df_train_))
label_train_.index = range(len(label_train_))

model = DBSCAN(eps=0.29,min_samples=10,)
model.fit(df_train_)

y_hat = list(model.labels_)
print(pd.Series(y_hat).value_counts())
print()
y_hat=pd.DataFrame(y_hat,columns=['cluster'],index= range(len(y_hat)))
y_hat[y_hat.cluster > -1] =0
y_hat[y_hat.cluster == -1] =1


from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
#print('roc:',roc_auc_score(label_test,y_pred))
print('precision:',precision_score(label_train_,y_hat['cluster']))
print('recall:',recall_score(label_train_,y_hat['cluster']))


print(' thrid cluster -----------------')

df_train__ = df_train_[y_hat.cluster == 1]
label_train__ = label_train_[y_hat.cluster == 1]

df_train__.index = range(len(df_train__))
label_train__.index = range(len(label_train__))

model = DBSCAN(eps=0.343,min_samples=50,)
model.fit(df_train__)

y_hat = list(model.labels_)
print(pd.Series(y_hat).value_counts())
print()
y_hat=pd.DataFrame(y_hat,columns=['cluster'],index= range(len(y_hat)))
y_hat[y_hat.cluster > -1] =0
y_hat[y_hat.cluster == -1] =1


from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
#print('roc:',roc_auc_score(label_test,y_pred))
print('precision:',precision_score(label_train__,y_hat['cluster']))
print('recall:',recall_score(label_train__,y_hat['cluster']))





c=1/0
FN = len(y_hat[label_train ==0][y_hat != -1])
TP = len(y_hat[label_train ==1][y_hat == -1])
TN = len(y_hat[label_train ==1][y_hat != -1])
FP = len(y_hat[label_train ==0][y_hat == -1])

print('recall of 1:', TP/(TP+FN))
print('precision of 1:',TP/(TP+FP))


