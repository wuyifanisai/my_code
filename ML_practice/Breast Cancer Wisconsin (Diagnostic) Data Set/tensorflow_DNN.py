import tensorflow as tf

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

'''
建立一个分类器，用以对样本进行分类
'''

df=pd.read_csv('E:\\Breast Cancer Wisconsin (Diagnostic) Data Set\\data_1.csv',header=0)
label=pd.read_csv('E:\\Breast Cancer Wisconsin (Diagnostic) Data Set\\label.csv',header=None)

###################################################### feature selecting ####################################
top_fea=10
model = RandomForestClassifier(random_state=1)
model.fit(df, label)
feature_imp = pd.DataFrame(model.feature_importances_, index=df.columns, columns=["importance"])
feat_rf = feature_imp.sort_values("importance", ascending=False).head(top_fea).index

#reduce features
features = np.hstack([feat_rf])
features = list(np.unique(features))
df=df[features]

############################ prepare the data ########################################

df=df.as_matrix()
#label=label.as_matrix()
label=np.array(label).reshape(1,569)[0]
df_train,df_test,label_train,label_test = df[:400],df[400:],label[:400],label[400:]

label_train = [ [n] for n in label_train ]
label_test = [ [n] for n in label_test ]
print(df_train)
	

#---------------------------------------------------------------------------------------------------------------

# 获取每一层的权值参数，并把权值参数得到的L2正则项的部分加入到损失集合中losses去
def get_weight(shape,lamb):
	var = tf.Variable(tf.random_normal(shape) , dtype = tf.float32)
	tf.add_to_collection('losses' , tf.contrib.layers.l2_regularizer(lamb)(var))
	return var

#神经网络输入数据存放处
x = tf.placeholder(tf.float32, shape=(None,top_fea))
y_ = tf.placeholder(tf.float32 , shape=(None,1))

data_size =400
#设定神经网络的批量训练参数
batch_size = 50

#设定每一层的神经元个数
layer = [top_fea,20,5,1]
n_layer = len(layer)

cur_layer = x #当前的神经层
input_dimension = layer[0]
all_weight=[]#存放所有的权值参数
all_bias=[]

#通过循环来构造每一层数据与权值的线性运算以及通过激活函数的计算
for i in range(1,n_layer):
	output_dimension = layer[i]
	weight = get_weight([input_dimension , output_dimension],0.01)
	bias = tf.Variable(tf.constant(0.01,shape = [output_dimension]))
	cur_layer = tf.nn.relu(tf.matmul(cur_layer , weight) + bias)
	all_weight.append(weight)
	all_bias.append(bias)
	input_dimension = layer[i]


#将损失函数加入losses
#loss = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(cur_layer,1e-10,1.0)))
entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(cur_layer , y_))
loss = entropy+tf.add_n(tf.get_collection('losses'))

X=df_train
Y=label_train

X1=df_test
Y1=label_test

#定义优化训练步
train_step = tf.train.AdamOptimizer(0.00002).minimize(loss)

#进行循环优化计算的过程的会话
with tf.Session() as sess:
	init_op = tf.initialize_all_variables()
	sess.run(init_op)
	print(sess.run(weight))

	steps = 1000
	for i in range(steps):
		for j in range(data_size//batch_size):
			start = j*batch_size
			end = j*batch_size+batch_size
			sess.run(train_step , feed_dict={x:X[start:end] , y_:Y[start:end]})
		if i%10==0:
			cur_loss = sess.run(loss , feed_dict = {x:X,y_:Y})
			print('it is ',i,' th iteration ,','loss is ',cur_loss)

	w=sess.run(all_weight)
	print(len(w))
	X1 = X1.astype(np.float32)
	
	#进行测试数据的测试
	a=X1
	for i in range(n_layer-1):
		a = sess.run(tf.nn.relu(tf.matmul(a,w[i])+all_bias[i]))
	print(a)
	for i in range(len(a)):
		if a[i]>0.5:
			a[i]=1.0
		else:
			a[i] = 0.0

	test_loss = tf.reduce_mean(tf.cast(tf.equal(Y1 , a) , tf.float32 ))

	for i in range(0,100):
		print(Y1[i] , a[i])
	print('LOSS OF TEST:',sess.run(test_loss))














