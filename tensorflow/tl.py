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
	names = ['tiger','cat']
	for name in names:
		os.makedirs('E://image_data//image_%s.txt'%name,exist_ok = True)
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
	image = skimage.io.imread(path)
	image = image/255.0
	short_edge = min(image.shape[:2])
	xx = int((image.shape[0] - short_edge)/2.0)
	yy = int((image.shape[1] - short_edge)/2.0)
	new_image = image[yy:yy + short_edge, xx:xx + short_edge] ## get the center part of a image
	resized_image = skimage.transform.resize(new_image, (224, 224))[None,:,:,:] ## transfrom the center part of image into 224x224 image
	return resized_image

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
			imag[name].append(resized_image)
			if len(imag[name]) == 400: #only use 400 iamges to get the train data
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
			self.data_dict = np.load(vgg16_npy_path, encoding= ='latin1').item()
		except:
			print('paramters of vgg is not found...')

	self.x = tf.placeholder(tf.float32,[None,224, 224, 3])
	self.y = tf.placeholder(tf.float32,[None,1])

	## covert the RGB to BGR
	red, green, blue = tf.split(axis = 3, num_or_size_split = 3, value = self.x * 255.0)
	bgr = tf.concat(axis =3,values=[blue - self.vgg_mean[0], green - self.vgg_meanp[1], blue - self.vgg_mean[2],])

	# prepare vgg16 conv layer to get the parameters from trained model
	conv11_out = self.conv_layer(bgr,'conv_11')
	conv12_out = self.conv_layer(conv11_out,'conv_12')
	pool1_out = self.max_pool(conv_12, 'pool1')

	conv21_out = self.conv_layer(pool1_out,'conv_21')
	conv22_out = self.conv_layer(conv21_out,'conv_22')
	pool2_out = self.max_pool(conv22_out, 'pool2')

	conv31_out = self.conv_layer(pool2_out,'conv_31')
	conv32_out = self.conv_layer(conv31_out,'conv_32')
	conv33_out = self.conv_layer(conv32_out,'conv_33')
	pool3_out = self.max_pool(conv33_out, 'pool3')

	conv41_out = self.conv_layer(pool3_out,'conv_41')
	conv42_out = self.conv_layer(conv41_out,'conv_42')
	conv43_out = self.conv_layer(conv42_out,'conv_43')
	pool4_out = self.max_pool(conv43_out, 'pool4')

	conv51_out = self.conv_layer(pool4_out,'conv_51')
	conv52_out = self.conv_layer(conv51_out,'conv_52')
	conv53_out = self.conv_layer(conv52_out,'conv_53')
	pool5_out = self.max_pool(conv53_out, 'pool5')

	# full connection layer
	self.flatten = tf.reshape(pool5_out, [-1, 7*7*512])
	self.full6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name = 'full6')
	self.out = tf.layers.dense(self.full6, 1, name = 'out')

	sess = tf.Session()

	if restore_form:
		saver = tf.train.Saver()
		saver.restore(self.sess, restore_form)

	else:
		self.loss = tf.losses.mean_squared_error(labels = self.y, predictions = self.out)
		self.train_op = tf.train.RMSPropOptimizer(0.001).minmize(self.loss)
		self.sess.run(tf.global_variables_initializer())


 
	def conv_layer(self, input, name):
		with tf.variable_scope(name):
			conv = tf.nn.conv2d(input, self.data_dict[name][0], [1,1,1,1], padding = 'SAME')
			out = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1])) 
			return out

	def max_pool(self, input, name):
		with tf.variable_scope(name)
			return tf.nn.max_pool(input , ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME')


	def train(self,x,y):
		loss, _ = self.sess.run([self.loss, self.train_op],{self.x:x, self.y:y})
		return loss

	def predict(self, paths):
		fig , axs = plt.subplots(1,2)
		for i , path in enumerate(paths):
			x = resize_image(path)
			length = self.sess.run(self.out, {self.x:x})

			axs[i].set_title('length:%.1f cm'%length)
			axs[i].imshow(x[0])
			axs[i].set_xticks(())
			axs[i].set_yticks(())
		plt.show()

	def save(self, path='e://transfer'):
		saver = tf.train.Saver()
		saver.save(self.sess, path , write_mate_graph= False)

def train():
	tigers_x, cats_x, tigers_y, cats_y=get_data()
	train_x = np.concatenate(tigers_x + cats_x, axis = 0)
	train_y = np.concatenate((tigers_y , cats_y), axis = 0)

	vgg = VGG16(vgg16_npy_path = 'E://transfer//vgg16.npy')
	print('vgg is bulit !')

	for i in range(100):
		loss  = vgg.train(train_x[np.random.randint(0,len(train_x),6)], train_y[np.random.randint(0,len(train_x),6)])
		print('it is step',i,' the loss is', loss)
	vgg.save()

def test():
	vgg = VGG16(vgg16_npy_path = 'E://transfer//vgg16.npy', restore_form = 'e://transfer')
	vgg.predict(['http://farm1.static.flickr.com/101/266907904_a927c86af8.jpg'
					,'http://farm3.static.flickr.com/2089/2146943800_c0a6d4606b.jpg'])

if __name__ = '__main__':
	image_load()
	train()
	test()







	





