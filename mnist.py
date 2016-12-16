#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 22:05:19 2016

Neural network for digit classification (MNIST) Kaggle competition
conv/conv/pool/dense/readout
Backend is Tensorflow

@author: dario
"""
# import the packages
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
#from keras.datasets import mnist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% 
# load the training data
df_train = pd.read_csv('digit-recognizer/train.csv')
y_train = df_train['label'].values
X_train = df_train.drop('label', axis=1).values

# load the test data
df_test = pd.read_csv('digit-recognizer/test.csv')
X_test = df_test.values

# number of samples
n_train, _ = X_train.shape
n_test, _ = X_test.shape

# dims of the pictures in pixels
height = 28
width = 28

#%% reshaping

# we have to preprocess the data into the right form
# in this case it must be explicitly stated that it's only grayscale
# otherwise for RGB pictures (n_train, height, width, 3)
X_train = X_train.reshape(n_train, height, width, 1).astype('float32')
X_test = X_test.reshape(n_test, height, width, 1).astype('float32')

# confirm that reshaping worked by plotting one digit example
pixels = X_train[9:10:,::,::]
pixels = pixels.reshape(height,width).astype('float32')
plt.imshow(pixels, cmap='gray')
plt.show()

# normalize from [0, 255] to [0, 1]
X_train /= 255
X_test /= 255

# numbers 0-9, so ten classes
n_classes = 10

# one-hot-encode the training labels
y_train = to_categorical(y_train, n_classes)

#%% Set up the neural network

# number of convolutional filters (neurons)
n_filters = 10

# convolution filter size
# i.e. we will use a n_conv x n_conv filter
n_conv = 5

# pooling window size
# i.e. we will use a n_pool x n_pool pooling window
n_pool = 2 # 2x2 reduces time for running model in half compared to 1x1

# build the neural network
model = Sequential()
# layer 1
model.add(Convolution2D(
        n_filters, n_conv, n_conv,
        
        # apply the filter to only full parts of the image
        # (i.e. do not "spill over" the border)
        # this is called a narrow convolution
        border_mode='valid',

        # we have a 28x28 single channel (grayscale) image
        # so the input shape should be (28, 28, 1)
        input_shape=(height, width, 1)
))
model.add(Activation('relu'))
# layer 2
model.add(Convolution2D(n_filters, n_conv, n_conv))
model.add(Activation('relu'))
# layer 3: pooling
model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))

model.add(Dropout(0.25))
# flatten the data for the 1D layers
model.add(Flatten())

# Dense(n_outputs)
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# softmax output layer gives probablity for each class
model.add(Dense(n_classes))
model.add(Activation('softmax'))

#%% compile model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#%%
from keras.utils.visualize_util import plot
plot(model, to_file='digit-recognizer/model.png')

#%% fit the model with validation
hist = model.fit(X_train, y_train, nb_epoch=5, validation_split=0.2, batch_size=32)
print(hist.history)

#save model to file
model.save('digit-recognizer/model_kaggle.h5')

#%%
# Plot accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%%
# Plot loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%% fit model without validation
#model.fit(X_train, y_train, nb_epoch=10, batch_size=32)

#%% evaluate on test data
#loss, accuracy = model.evaluate(X_test, y_test)
#print('loss:', loss)
#print('accuracy:', accuracy)

#%% predict
#model = load_model('digit-recognizer/model_kaggle.h5')

y_predict = model.predict(X_test, batch_size=32)
df_predict = pd.DataFrame(y_predict) # each sample has 10 features
df_predict = df_predict.idxmax(axis=1) # each sample has 1 feature (softmax)

df_submit = pd.DataFrame()
df_submit['ImageId'] = np.arange(1,28001)
df_submit['Label'] = df_predict.astype(int)
df_submit.to_csv('digit-recognizer/submission.csv', index=False)
