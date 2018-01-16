
	import numpy as np

	from keras.models import Sequential

	from keras.layers import Dense

	import matplotlib.pyplot as plt



	# create some data for X , Y------------

	X = np.linspace(-1,1,100)

	np.random.shuffle(X)

	Y = 0.5*X-0.35 + np.random.normal(0,0.05,(100,))

	plt.scatter(X,Y)

	plt.show()


	# get the train and test data-----------

	x_train,y_train = X[:70] , Y[:70]

	x_test,y_test = X[70:] , Y[70:]


	# build the model ---------

	model = Sequential()

	model.add(Dense(output_dim = 1 , input_dim = 1))

	model.compile(loss = 'mse' , optimizer = 'sgd')

	# training ------------

	print('training....')

	cost_list =[]

	for step in range(500):

	 cost = model.train_on_batch(x_train,y_train)

	 if step%5==0:

	 print('training cost:',cost)

	 cost_list.append(cost)

	plt.plot(cost_list)

	plt.show()

	# testing ------------

	print('testing.....')

	cost = model.evaluate(x_test,y_test)

	print('test cost',cost)

	w,b = model.layers[0].get_weights()

	print('weights is %f , biase is %f'%(w,b))




	# plotting the result of test--------------------


	y_pred = model.predict(x_test)


	plt.plot(x_test,y_pred)


	plt.scatter(x_test,y_test)


	plt.show()


	



	以下是拟合一个二次函数的keras模型


	import numpy as np


	from keras.models import Sequential


	from keras.layers import Dense


	import matplotlib.pyplot as plt


	from mpl_toolkits.mplot3d import Axes3D


	



	# create some data for X , Y------------


	X = np.linspace(-1,1,500)


	np.random.shuffle(X)


	Y = 0.5*X*X-0.35*X +1.1 + np.random.normal(0,0.05,(500,))


	#plt.scatter(X,Y)


	#plt.show()


	



	# get the train and test data-----------


	x_train,y_train = X[:400] , Y[:400]


	x_test,y_test = X[400:] , Y[400:]


	



	# build the model ---------


	model = Sequential()


	model.add(Dense(output_dim = 2 , input_dim = 1))


	model.add(Dense(output_dim = 5 , input_dim = 2,activation = 'relu'))


	model.add(Dense(output_dim = 1 , input_dim = 5,activation = 'relu'))


	model.compile(loss = 'mse' , optimizer = 'sgd')


	



	# training ------------


	print('training....')


	cost_list =[]


	for step in range(2000):


	 cost = model.train_on_batch(x_train,y_train)


	 if step%100==0:


	 print('training cost:',cost)


	 cost_list.append(cost)


	plt.plot(cost_list)


	plt.show()


	# testing ------------


	print('testing.....')


	cost = model.evaluate(x_test,y_test)


	print('test cost',cost)


	w,b = model.layers[0].get_weights()


	#print('weights is %f , biase is %f'%(w,b))

	# plotting the result of test--------------------


	y_pred = model.predict(x_test)


	plt.scatter(x_test,y_pred,c='r')


	plt.scatter(x_test,y_test,c='b')


	plt.show()


		
import numpy as np

from keras.models import Sequential

from keras.layers import Dense

import matplotlib.pyplot as plt



# create some data for X , Y------------

X = np.linspace(-1,1,300)

np.random.shuffle(X)

Y = 0.5*X**3-0.35*X + np.random.normal(0,0.03,(300,))

#plt.scatter(X,Y)

#plt.show()



# get the train and test data-----------

x_train,y_train = X[:200] , Y[:200]

x_test,y_test = X[200:] , Y[200:]



# build the model ---------

model = Sequential()

model.add(Dense(output_dim = 5 , input_dim = 1,activation = 'relu'))

model.add(Dense(output_dim = 8 , activation = 'relu'))

#model.add(Dense(output_dim = 20 , activation = 'relu'))

#model.add(Dense(output_dim = 4 , activation = 'relu'))

model.add(Dense(output_dim = 1,activation = 'linear'))

model.compile(loss = 'mse' , optimizer = 'sgd')



# training ------------

print('training....')

cost_list =[]

for step in range(40000):

    cost = model.train_on_batch(x_train,y_train)

    if step%100==0:

        print('training cost--:',step,'--',cost)

        cost_list.append(cost)

plt.plot(cost_list)

plt.show()

# testing ------------

print('testing.....')

cost = model.evaluate(x_test,y_test)

print('test cost ',cost)

'''

w1,b1 = model.layers[0].get_weights()

w2,b2 = model.layers[1].get_weights()

w3,b3 = model.layers[2].get_weights()

print('weights1 is  , biase1 is',(w1,b1))

print('weights2 is  , biase2 is',(w2,b2))

print('weights3 is  , biase3 is',(w3,b3))

'''



# plotting the result of test--------------------

y_pred = model.predict(x_test)

plt.scatter(x_test,y_pred,c = 'r')

plt.scatter(x_test,y_test,c = 'b')

plt.show()





					

						

					

				

			

		

	
