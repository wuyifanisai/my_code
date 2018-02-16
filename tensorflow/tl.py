'''
This is a example of transfer learning applied in CNN for imagination classification
Fine tune a VGG16 model based on a trained version foe classification 
and train the model to become a regerssor to output a length of a cat or tiger in the picture

we think the length of cat should come from the normal disturibution (40,8)
and the length of tiger should come from the normal distrubution (100,30)
so the model from fine tune based on the VGG16 should predict that it is a cat or tiger firstly and give 
the length of the animal from the distribution 
'''
from urllib.request import urlretrieve
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform

######################### function to download the image of tiger and cat from internet ###################
def image_load():
	
	#names = ['tiger','cat','dog']
	names =['dog']
	for name in names:
		os.makedirs('E://image_data//%s'%name,exist_ok = True)
		with open('E://image_data_urls//%s.txt'%name,'r') as file:
			urls = file.readlines()
			len_urls = len(urls)
			for i, url in enumerate(urls):
				print('******************************')
				print(url)
				try:
					urlretrieve(url.strip(), 'E://image_data//%s//%s'%(name, url.strip().split('/')[-1]))
					print('%s %i/%i'%(name, i, len_urls))
				
				except:
					print('%s %i/%i'%(name, i, len_urls), 'no image')

################## function to dealing the image into 244,244 size ############################
def resize_image(path):
	try:
		image = skimage.io.imread(path)
		image = image/255.0
		short_edge = min(image.shape[:2])
		xx = int((image.shape[0] - short_edge)/2.0)
		yy = int((image.shape[1] - short_edge)/2.0)
		new_image = image[yy:yy + short_edge, xx:xx + short_edge] ## get the center part of a image
		resized_image = skimage.transform.resize(new_image, (224, 224))[None,:,:,:] ## transfrom the center part of image into 224x224 image
		print('a image is resized !',resized_image.shape,'-----------')
		if resized_image.shape != (1,224,224,3):
			print('*******************************************************************************************************')
			print(resized_image.shape)
			print(path)
		return resized_image
	except:
		print('error with skimage.io.imread')

#################### get the train data and label ########################
def get_data():
	image ={'tiger':[], 'cat':[]}
	for name in image.keys():
		dir = 'E://image_data//' + name
		for file in os.listdir(dir):
			if not file.lower().endswith('.jpg'):
				continue
			try:
				resized_image = resize_image(os.path.join(dir, file))
			except OSError:
				continue

			if resized_image is not None:
				if resized_image.shape == (1,224,224,3):
					image[name].append(resized_image)

			if len(image[name]) == 400: #only use 400 iamges to get the train data
				break
	# get the label of length for tigers and cats
	tigers_label = np.maximum(20,np.random.rand(len(image['tiger']) , 1)*30 + 100)
	cats_label = np.maximum(10,np.random.rand(len(image['cat']) , 1)*8 + 40)
	return image['tiger'], image['cat'], tigers_label, cats_label


############################## define the class of VGG 16 ###########################
class VGG16(object):
	vgg_mean=[103.939, 116.779, 123.68]

	def __init__(self, vgg16_npy_path = None, restore_form = None):
		# prepare trained parameters of VGG 16 
		try:
			self.data_dict = np.load(vgg16_npy_path, encoding ='latin1').item()
			print('the vgg is downloaded !')
		except:
			print('paramters of vgg is not found...')

		self.x = tf.placeholder(tf.float32,[None,224, 224, 3])
		self.y = tf.placeholder(tf.float32,[None,1])

		## covert the RGB to BGR
		red, green, blue = tf.split( split_dim = 3,  num_split = 3, value = self.x * 255.0)
		bgr = tf.concat(concat_dim = 3,values=[blue - self.vgg_mean[0], green - self.vgg_mean[1], red - self.vgg_mean[2],])

		# prepare vgg16 conv layer to get the parameters from trained model
		conv11_out = self.conv_layer(bgr,'conv1_1')
		conv12_out = self.conv_layer(conv11_out,'conv1_2')
		pool1_out = self.max_pool(conv12_out, 'pool1')

		conv21_out = self.conv_layer(pool1_out,'conv2_1')
		conv22_out = self.conv_layer(conv21_out,'conv2_2')
		pool2_out = self.max_pool(conv22_out, 'pool2')	

		conv31_out = self.conv_layer(pool2_out,'conv3_1')
		conv32_out = self.conv_layer(conv31_out,'conv3_2')
		conv33_out = self.conv_layer(conv32_out,'conv3_3')
		pool3_out = self.max_pool(conv33_out, 'pool3')

		conv41_out = self.conv_layer(pool3_out,'conv4_1')
		conv42_out = self.conv_layer(conv41_out,'conv4_2')
		conv43_out = self.conv_layer(conv42_out,'conv4_3')
		pool4_out = self.max_pool(conv43_out, 'pool4')

		conv51_out = self.conv_layer(pool4_out,'conv5_1')
		conv52_out = self.conv_layer(conv51_out,'conv5_2')
		conv53_out = self.conv_layer(conv52_out,'conv5_3')
		pool5_out = self.max_pool(conv53_out, 'pool5')

		# full connection layer
		self.flatten = tf.reshape(pool5_out, [-1, 7*7*512])

		self.out =self.build_layers(   
							self.flatten,
							256,
							w_initializer = tf.random_normal_initializer(mean=0,stddev =0.3),
							b_initializer = tf.random_normal_initializer(mean=0,stddev =0.3),
							)

		#self.full6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name = 'full6')
		#self.out = tf.layers.dense(self.full6, 1, name = 'out')


		self.sess = tf.Session()

		if restore_form:
			self.loss = tf.reduce_mean(tf.pow((self.y-self.out),2))
			#self.loss = tf.losses.mean_squared_error(labels = self.y, predictions = self.out)
			self.train_op = tf.train.RMSPropOptimizer(0.002).minimize(self.loss)
			saver = tf.train.Saver()
			saver.restore(self.sess, restore_form)


		else:
			self.loss = tf.reduce_mean(tf.pow((self.y-self.out),2))
			#self.loss = tf.losses.mean_squared_error(labels = self.y, predictions = self.out)
			self.train_op = tf.train.RMSPropOptimizer(0.002).minimize(self.loss)
			self.sess.run(tf.global_variables_initializer())


 
	def conv_layer(self, input, name):
		with tf.variable_scope(name):
			conv = tf.nn.conv2d(input, self.data_dict[name][0], [1,1,1,1], padding = 'SAME')
			out = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1])) 
			return out

	def max_pool(self, input, name):
		with tf.variable_scope(name):
			return tf.nn.max_pool(input , ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME')

	def build_layers(self,s, n_l1, w_initializer, b_initializer, train=True):
			with tf.variable_scope('full'):	
				w1 = tf.get_variable('w1', [7*7*512, n_l1], initializer=w_initializer, trainable=train)
				b1 = tf.get_variable('b_1', [1, n_l1], initializer=b_initializer,  trainable=train)
				l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
			with tf.variable_scope('out'):
				w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer,   trainable=train)
				b2 = tf.get_variable('b_2', [1, 1], initializer=b_initializer,   trainable=train)
				out = tf.matmul(l1, w2) + b2
			return out

	def train(self,x,y):
		loss, _ = self.sess.run([self.loss, self.train_op],{self.x:x, self.y:y})
		return loss

	def predict(self, paths):
		fig , axs = plt.subplots(1,2)
		for i , path in enumerate(paths):    
			x = resize_image(path)
			print(path)
			print(x.shape)
			length = self.sess.run(self.out, {self.x:x})

			axs[i].set_title('length:%.1f cm'%length)
			axs[i].imshow(x[0])
			axs[i].set_xticks(())
			axs[i].set_yticks(())
		plt.show()

	def save(self, path='e:transfer\\model.ckpt'):
		saver = tf.train.Saver()
		saver.save(self.sess, path)

def train():

	vgg = VGG16(vgg16_npy_path = 'E://vgg16.npy',restore_form = 'e:transfer\\.\\model.ckpt')
	print('vgg is bulit !')

	tigers_x, cats_x, tigers_y, cats_y=get_data() 
	print(type(tigers_x),len(tigers_x))
	print(type(cats_x),len(cats_x))

	# plot fake length distribution
	plt.hist(tigers_y, bins=20, label='Tigers')
	plt.hist(cats_y, bins=10, label='Cats')
	plt.legend()
	plt.xlabel('length')
	#plt.show()

	#tigers_x is a list, tiger_y is a array
	train_x = np.concatenate( tigers_x+cats_x, axis = 0)
	train_y = np.concatenate((tigers_y , cats_y), axis = 0)
	print(type(train_x), train_x.shape)
	print(type(train_y),train_y.shape)

	for i in range(150): # number of iteratins for train
		print('train step ==>',i)
		id_batch = list(np.random.randint(0,len(train_x),6))   

		loss  = vgg.train(  train_x[id_batch],  train_y[id_batch])

		print('it is step',i,' the loss is', loss)
	
	vgg.save()

def test():
	vgg = VGG16(vgg16_npy_path = 'E://vgg16.npy', restore_form = 'e:transfer\\.\\model.ckpt')
	vgg.predict([
					'http://img.zcool.cn/community/0113be56640d5a32f8754573d17fa6.jpg'
				])

if __name__ == '__main__':
	pass
	#image_load()
	train() 
	#test()  







	





