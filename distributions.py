# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:34:42 2020

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
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
import matplotlib as plt
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

keras.initializers.glorot_normal(seed=42)

train_path='C:/Users/Panagiotis Gkanos/Desktop/dataset/400X/train'
valid_path='C:/Users/Panagiotis Gkanos/Desktop/dataset/400X/valid'
test_path='C:/Users/Panagiotis Gkanos/Desktop/dataset/400X/test'

train_batches=ImageDataGenerator().flow_from_directory(train_path,
                                                      target_size=[400,400],
                                                      classes=['malignant','benign'],
                                                      class_mode='categorical',batch_size=1164,seed=7)

trainX,trainy=next(train_batches)
mean=trainX.mean()
std=trainX.std()
def myfunc(image):
    image=np.array(image)
    con_image=(image-mean)/std
    return con_image
    
    
    
    
width, height, channels = trainX.shape[1], trainX.shape[2], 3
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
# report pixel means and standard deviations
print('Statistics train=%.3f (%.3f)' % (trainX.mean(), trainX.std()))
# create generator that centers pixel values
datagen = ImageDataGenerator(preprocessing_function=myfunc)
# calculate the mean on the training dataset
datagen.fit(trainX)
#print( sum(datagen.mean[0][0])/3, sum(datagen.std[0][0]/3))
# demonstrate effect on a single batch of samples
iterator = datagen.flow_from_directory(train_path,classes=['malignant','benign'],
                                                      class_mode='categorical',batch_size=20,seed=7,target_size=[400,400])
# get a batch
batchX, batchy = iterator.next()
# pixel stats in the batch
print(batchX.shape, batchX.mean(), batchX.std())
# demonstrate effect on entire training dataset
iterator2 = datagen.flow_from_directory(test_path,classes=['malignant','benign'],
                                                      class_mode='categorical',batch_size=20,seed=7,target_size=[400,400])
# get a batch
batchX, batchy = iterator2.next()
# pixel stats in the batch
print(batchX.shape, batchX.mean(), batchX.std())





    
    
    



