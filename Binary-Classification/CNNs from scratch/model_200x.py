# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:05:12 2019

@author: Panagiotis Gkanos
"""
import numpy as np 
import tensorflow as tf
from numpy.random import seed
seed(1)

tf.compat.v1.set_random_seed(2)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
import os
os.environ['KERAS_BACKEND']='tensorflow'


import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
import matplotlib as plt
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

init=keras.initializers.glorot_normal(seed=42)

train_path='C:/Users/Panagiotis Gkanos/Desktop/dataset/200X/train'
train_batches=ImageDataGenerator().flow_from_directory(train_path,
                                                      target_size=[400,400],
                                                      classes=['malignant','benign'],
                                                      class_mode='categorical',batch_size=1276,seed=7)
#get a batch and calc std and mean 
trainX,trainy=next(train_batches)
mean=trainX.mean()
std=trainX.std()
#z-score func
def myfunc(image):
    image=np.array(image)
    con_image=(image-mean)/std
    return con_image

#get the images in right shape
width, height, channels = trainX.shape[1], trainX.shape[2], 3
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
#initialize imgdatagen with prepeoc func to cala z-score
datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=10,preprocessing_function=myfunc)
#calc z-score according to train set
'''
datagen.fit(trainX)
#get train, valid, test data and use preproc func to normalize 
train_batches = datagen.flow_from_directory(train_path,
                                            target_size=[400,400],
                                            classes=['malignant','benign'],
                                            class_mode='categorical', batch_size=40,seed=7)

valid_path='C:/Users/Panagiotis Gkanos/Desktop/dataset/40X/valid'
valid_batches=datagen.flow_from_directory(valid_path,
                                            target_size=[400,400],
                                            classes=['malignant','benign'],
                                            class_mode='categorical', batch_size=20,seed=7)
'''
test_path='C:/Users/Panagiotis Gkanos/Desktop/dataset/200X/test'
test_datagen=ImageDataGenerator(preprocessing_function=myfunc)
test_batches=test_datagen.flow_from_directory(test_path,
                                            target_size=[400,400],
                                            classes=['malignant','benign'],
                                            class_mode='categorical', batch_size=20,seed=7)




model=Sequential()

model.add(Conv2D(16,(3,3),strides=2,padding='same',input_shape=(400,400,3),kernel_initializer=init))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(16,(3,3),strides=1,padding='same',kernel_initializer=init))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(16,(3,3),strides=1,padding='same',kernel_initializer=init))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))



model.add(Conv2D(32,(3,3),strides=1,padding='same',kernel_initializer=init))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),strides=1,padding='same',kernel_initializer=init))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),strides=1,padding='same',kernel_initializer=init))
model.add(Activation('relu'))
model.add(BatchNormalization())
#model.add(Conv2D(64,(7,7),padding='same'))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))



model.add(Conv2D(64,(3,3),strides=1,padding='same',kernel_initializer=init))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),strides=1,padding='same',kernel_initializer=init))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))



model.add(Conv2D(128,(3,3),strides=1,padding='same',kernel_initializer=init))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))



model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
#model.add(Dense(32,activation='relu'))
model.add(Dropout(0.3))



model.add(Dense(2,activation='softmax'))
model.summary()
'''
history=model.fit_generator(train_batches,steps_per_epoch=20 ,validation_data=valid_batches,
                    validation_steps=8 ,epochs=100)
'''
filepath='C:/Users/Panagiotis Gkanos/Desktop/X200/3x16-3x32-2x64-1x128-256-128/BN+DO+z-score+he+HF+VF+RR30+BR/sec_100min_weights.best.hdf5'


model.load_weights(filepath)
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

model.evaluate(test_batches)
'''
def plot_loss(history):
    train_loss=history.history['loss']
    val_loss=history.history['val_loss']
    x=list(range(1,len(val_loss)+1))
    plt.plot(x,val_loss,color='red',label='validation loss')
    plt.plot(x,train_loss,label='training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.show()

    
def plot_accuracy(history):
    train_acc=history.history['acc']
    val_acc=history.history['val_acc']
    x=list(range(1,len(val_acc)+1))
    plt.plot(x,val_acc,color='red',label='validation acc')
    plt.plot(x,train_acc,label='training acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    plt.show()

plot_loss(history)

plot_accuracy(history)
'''









