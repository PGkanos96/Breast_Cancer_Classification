# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:43:00 2020

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


import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
import itertools
import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input


#get train set to calc mean and std for z-score
train_path='C:/Users/Panagiotis Gkanos/Desktop/dataset/400X/train'

#z-score func
def myfunc(image):
  #  image=np.array(image)
   # con_image=(image-mean)/std
    con_image = preprocess_input(image)
    return con_image


#z-score func


#initialize imgdatagen with prepeoc func to cala z-score
datagen = ImageDataGenerator(rescale=1/255,horizontal_flip=True,vertical_flip=True,preprocessing_function=myfunc)

#get train, valid, test data and use preproc func to normalize 

train_batches = datagen.flow_from_directory(train_path,
                                            target_size=[250,250],
                                            classes=['malignant','benign'],
                                            class_mode='categorical', batch_size=40,seed=7)

valid_path='C:/Users/Panagiotis Gkanos/Desktop/dataset/400X/valid'
valid_batches=datagen.flow_from_directory(valid_path,
                                            target_size=[250,250],
                                            classes=['malignant','benign'],
                                            class_mode='categorical', batch_size=40,seed=7)

test_path='C:/Users/Panagiotis Gkanos/Desktop/dataset/400X/test'
test_datagen=ImageDataGenerator(rescale = 1./255,preprocessing_function=myfunc)
test_batches=test_datagen.flow_from_directory(test_path,
                                            target_size=[250,250],
                                            classes=['malignant','benign'],
                                            class_mode='categorical', batch_size=10,seed=7)

vgg16_model = keras.applications.vgg16.VGG16(input_shape=(250,250,3), weights='imagenet', include_top=False)

model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)
    
model.layers.pop()
model.summary()
for layer in model.layers:
    layer.trainable = False


model.add(Flatten())


model.add(Dense(2,activation='softmax'))
model.summary()
#adam =keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

#filepath='C:/Users/Panagiotis Gkanos/Desktop/X40/3x16-3x32-1x64-1x128-256-128/6.0/2min_weights.best.hdf5'
filepath='C:/Users/Panagiotis Gkanos/Desktop/tranfser_400.hdf5'

#model.load_weights(filepath)
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])


checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
learn_control = ReduceLROnPlateau(monitor='accuracy', patience=5,
                                  verbose=1,factor=0.1, min_lr=1e-10)
callbacks_list = [checkpoint,learn_control]
history=model.fit_generator(train_batches,steps_per_epoch=20 ,validation_data=valid_batches,
                            callbacks=callbacks_list,
                            validation_steps=15 ,epochs=100)
loss, acc = model.evaluate_generator(test_batches,val_samples=20)
print(loss, acc)

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







