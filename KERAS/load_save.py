
####  while saving and loading ,a model named h5py is needed, so type "sudo pip3 install h5py " in terminal....


from keras.models import Sequential

from keras.layers import Dense

from keras.models import load_model

import matplotlib.pyplot as plt

import numpy as np


 # create some data


X = np.linspace(-1,1,500)

np.random.shuffle(X)

Y = 0.5 * X + 2 + np.random.normal(0,0.05,( 500, ))


# create train and test data

x_train , y_train  = X[:400] , Y[:400]

x_test , y_test = X[400:] , Y[400:]


# CREATE MODEL ----

model = Sequential()

model.add(Dense(input_dim = 1 , output_dim = 1 ,activation = 'relu'))

model.compile(loss = 'mse' , optimizer = 'sgd')


#train

cost_list=[]

for step in range(500):

    cost = model.train_on_batch(x_train , y_train)

    cost_list.append(cost)

    if step%5 ==0:

        print('cost ====>',cost)

plt.plot(cost_list)

plt.show()

#save


print('test before saving model...')

print(model.predict(x_test[:2]))

model.save('my_model.h5')

del model


#load

model = load_model('my_model.h5')

print('test after loading model...')

print(model.predict(x_test[:2]))

print('test accuracy :',model.evaluate(x_test , y_test))

plt.scatter(x_test,model.predict(x_test),c='r')

plt.scatter(x_test , y_test,c='b')

plt.show()



#another way to save and load

# only save the weights of model , the structure of model is not saved

model.save_weights('my_model_weights.h5')

# to bulid a new model whose structure is same as before ....

new_model.load_weights('my_model_weights.h5')


# a method to save model structure only ,well-trained weights is not saved

from keras import model_from_json

json_string = model.to_json()

new_mode = model_from_json(json_string)
