# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:59:29 2019

@author: P_Gkanos
"""
import os
os.environ['KERAS_BACKEND']='theano'
import numpy as np 
import keras
from keras.metrics import categorical_crossentropy
import itertools
import glob
from shutil import copy2



#'C:/Users/P_Gkanos/Desktop/breast/benign'

img_zoom=['40','100','200','400']
#ITERATE OVER ZOOMS GET IMAGES SEPERATLY AND COPY THEM TO DIFFERENT DIRECTORIES
for i in range(len(img_zoom)):    
    ben_path=glob.glob('C:/Users/Panagiotis Gkanos/Desktop/breast/benign/**/*-'+img_zoom[i]+'-*.*', recursive=True)
    mal_path=glob.glob('C:/Users/Panagiotis Gkanos/Desktop/breast/malignant/**/*-'+img_zoom[i]+'-*.*', recursive=True)
    
    for f in range(len(ben_path)):
        copy2(ben_path[f], 'C:/Users/Panagiotis Gkanos/Desktop/dataset/'+img_zoom[i]+'X/benign')
    
    for j in range(len(mal_path)):
        copy2(mal_path[j], 'C:/Users/Panagiotis Gkanos/Desktop/dataset/'+img_zoom[i]+'X/malignant')




print(ben_path[1])
























"""
train_path='sample/train'
valid_path='\Desktop\breast\benign\SOB\adenosis\SOB_B_A_14-22549G\40X'
test_path='\Desktop\breast\benign\SOB\adenosis\SOB_B_A_14-22549CD\40X'

#train_path1='breast\malignant\SOB\ductal_carcinoma\SOB_M_DC_14-2523\40X'
#valid_path1='breast\malignant\SOB\ductal_carcinoma\SOB_M_DC_14-2773\40X'
#test_path1='breast\malignant\SOB\ductal_carcinoma\SOB_M_DC_14-2980\40X'

train_baches=ImageDataGenerator().flow_from_directory(train_path, target_size=[200,200], classes=['malignant','benign'], batch_size=10)

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
            
      
    
    
    
imgs,labels=next(train_baches)
plots(imgs,titles=labels)

print(len(imgs))
"""