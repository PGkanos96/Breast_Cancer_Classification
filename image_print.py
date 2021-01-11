# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:40:06 2020

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
import matplotlib as plt
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import cv2


train_path='C:/Users/Panagiotis Gkanos/Desktop/dataset/400X/train'

#train_path1='breast\malignant\SOB\ductal_carcinoma\SOB_M_DC_14-2523\40X'
#valid_path1='breast\malignant\SOB\ductal_carcinoma\SOB_M_DC_14-2773\40X'
#test_path1='breast\malignant\SOB\ductal_carcinoma\SOB_M_DC_14-2980\40X'

train_batches=ImageDataGenerator().flow_from_directory(train_path, target_size=[400,400], classes=['malignant','benign'], batch_size=1276,shuffle=False,seed=7)
trainX,trainy=next(train_batches)


mean=trainX.mean()
std=trainX.std()
#z-score func
def myfunc(image):
    image=np.array(image)
    con_image=(image-mean)/std
    return con_image

#width, height, channels = trainX.shape[1], trainX.shape[2], 3
#trainX = trainX.reshape((trainX.shape[0], width, height, channels))
print('Statistics train=%.3f (%.3f)' % (trainX.mean(), trainX.std()))
# create generator that centers pixel values
datagen = ImageDataGenerator(
                             #samplewise_std_normalization=True,samplewise_center=True,
                             #zca_whitening=True
                             #brightness_range=[0.5,1.0],
                             preprocessing_function=myfunc
                             )
#calc z-score according to train set
datagen.fit(trainX)
print((datagen.mean, datagen.std))
t_batches=datagen.flow_from_directory(train_path, target_size=[400,400], classes=['malignant','benign'], batch_size=100,seed=7)


def plots(ims,figsize=(12,6),rows=1,interp=False,titles=None):
    if type(ims[0]) is np.ndarray:
        ims=np.array(ims).astype(np.uint8)
        if (ims.shape[-1]!=3):
            ims=ims.transpose((0,2,3,1))
    f=plt.figure(figsize=figsize)
    cols=len(ims)//rows if len(ims)%2==0 else len(ims)//rows+1
    for i in range(len(ims)):
        sp=f.add_subplot(rows,cols,i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
            
      
    
    
    
imgs,labels=next(t_batches)
# alternative way to find histogram of an image 
plt.hist(imgs.ravel(),100,[-3,5]) 
plt.show() 
print(imgs.mean(),imgs.std())
print(len(imgs))

#plots(imgs)
