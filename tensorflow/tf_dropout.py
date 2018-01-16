import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

#load data from sklearn------------------------
digits = load_digits()
X = digits.data
y = digits.target
y=LabelBinarizer().fit_transform(y)

# prepare train data and test data------------
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

# define fnuction called 'add_layer'----------
def add_layer(inputs , input_fea_num , output_fea_num , layer_name , activation = None ):
    Weights = tf.Variable(tf.random_normal([input_fea_num,output_fea_num],0,0.1))
    biase = tf.Variable(tf.zeros([1,output_fea_num]) + 0.1)
    Wx_plus_b = tf.matmul(inputs , Weights) + biase
    Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_pro)   
 #通过tf.nn.dropout 设置keep_pro来实现神经元的dropout功能

    if activation == None:
        output =  Wx_plus_b
    else:
        output =  activation(Wx_plus_b)

    #tf.histogram_summary(layer_name+'/outputs', output)
    return output

# define placeholder ------------
keep_pro = tf.placeholder(tf.float32)   #给dropout的比例定义一个placeholder

xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32,[None,10])

# add layer-------
output1 = add_layer(xs , 64 , 100 , 'layer1', activation = tf.nn.tanh)
pre = add_layer(output1 , 100 , 10 , 'layer2', activation= tf.nn.softmax)

# loss ----------
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(pre) , reduction_indices=[1]))
#tf.scalar_summary('loss' , cross_entropy)
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cross_entropy)

# session -----------
sess = tf.Session()
#merged = tf.merge_all_summaries()
#train_writer = tf.train.SummaryWriter('E:\\',sess.graph)
#test_writer = tf.train.SummaryWriter('E:\\',sess.graph)

# RUN ---------------
sess.run(tf.initialize_all_variables())
loss_train=[]
loss_test=[]
for step in range(500):
    sess.run(train_step , {xs:x_train,ys:y_train , keep_pro : 0.6}) 
    # 在神经网络训练时候给dropout输入一个小于1的比例，表示保留下来的神经元的比例 
    if step % 10 ==0:
        cost_train = sess.run(cross_entropy,{xs:x_train , ys:y_train, keep_pro : 1.0})
        cost_test = sess.run(cross_entropy,{xs:x_test , ys:y_test , keep_pro : 1.0})
#在用神经网络进行预测输出的时候，dropout输入 1.0 ，因为用神经网络预测输出的时候所有神经元都打开

        loss_train.append(cost_train)
        loss_test.append(cost_test)

plt.plot(loss_train,c='r')
plt.plot(loss_test,c='b')
plt.show()


        #train_result = sess.run(merged ,{xs:x_train,ys:y_train , keep_pro : 1})
        #test_result = sess.run(merged,{xs:x_test,ys:y_test , keep_pro : 1})

        #train_writer.add_summary(train_result,step)
        #test_writer.add_summary(test_result,step)


