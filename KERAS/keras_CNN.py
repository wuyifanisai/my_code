import numpy as np
from keras.models import Sequential 
from keras.utils import np_utils
from keras.layers import Dense , Activation
from keras.layers import Dense ,Activation , Convolution2D , MaxPooling2D, Flatten
from keras.datasets import mnist
from keras.optimizers import Adam

# create the train data and test data ----
(x_train , y_train ) , (x_test , y_test)= mnist.load_data()

# data preprocessing ---------
x_train = x_train.reshape(-1, 1, 28 ,28)
x_test = x_test.reshape(-1, 1, 28, 28)

y_train = np_utils.to_categorical(y_train , 10)
y_test = np_utils.to_categorical(y_test , 10)


# bulid your CNN model -------------
model = Sequential()
model.add(Convolution2D( filters = 32, nb_row = 5, nb_col = 5, border_mode = 'same', input_shape = (1, # number of channel
		28,28)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2) , strides = (2,2) , border_mode = 'same',))

model.add(Convolution2D(64,5,5,border_mode = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2) , border_mode = 'same',))


model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr = 1e-4)

model.compile(optimizer = adam , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

# training --------------------------
print('trainging -------')
model.fit(x_train , y_train ,  nb_epoch =1 , batch_size =  32,)

# test ------------------------
print('testing =======')
loss , accuracy = model.evaluate(x_test , y_test)
print('loss of test is %f , accuarcy of test is %f !'%(loss,accuracy))
