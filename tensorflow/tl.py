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

	





