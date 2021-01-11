# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 22:00:28 2019

@author: Panagiotis Gkanos
"""
import os
os.environ['KERAS_BACKEND']='theano'
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image



def datasplit(data_path,store_path,myclass,batch):
    print(data_path,store_path,myclass)
    
    data=ImageDataGenerator().flow_from_directory(data_path, target_size=[460,700], classes=[myclass], batch_size=batch)

    X,Y=next(data)
    X_tr,X_test,y_tr,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    X_train,X_valid,y_train,y_valid=train_test_split(X_tr,y_tr,test_size=0.2,random_state=42)



    X_train=np.array(X_train).astype(np.uint8)
    X_valid=np.array(X_valid).astype(np.uint8)
    X_test=np.array(X_test).astype(np.uint8)



    for k in range(int(len(X_train))):
        train=Image.fromarray(X_train[k],'RGB')
        p=train.save(store_path[0] +str(k)+'.png')
    
    
    for k in range(int(len(X_valid))):
        valid=Image.fromarray(X_valid[k],'RGB')
        p=valid.save(store_path[1] +str(k)+'.png')
    
    for k in range(int(len(X_test))):
        test=Image.fromarray(X_test[k],'RGB')
        p=test.save(store_path[2] +str(k)+'.png')


 
data_path='C:/Users/Panagiotis Gkanos/Desktop/dataset/400X'
myclass=['benign','malignant']
batch=[625,1370]
ben_store_path=['C:/Users/Panagiotis Gkanos/Desktop/dataset/400X/train/benign/ben_',
                'C:/Users/Panagiotis Gkanos/Desktop/dataset/400X/valid/benign/ben_',
                'C:/Users/Panagiotis Gkanos/Desktop/dataset/400X/test/benign/ben_']
mal_store_path=['C:/Users/Panagiotis Gkanos/Desktop/dataset/400X/train/malignant/mal_',
                'C:/Users/Panagiotis Gkanos/Desktop/dataset/400X/valid/malignant/mal_',
                'C:/Users/Panagiotis Gkanos/Desktop/dataset/400X/test/malignant/mal_']

datasplit(data_path,ben_store_path,myclass[0],batch[0])
datasplit(data_path,mal_store_path,myclass[1],batch[1])


































"""
def plots(ims,figsize=(20,10),rows=1,interp=False,titles=None):
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

plots(X_train,titles=y_train)
k=ig.save('C:/Users/Panagiotis Gkanos/Desktop/gay.jpg')"""
