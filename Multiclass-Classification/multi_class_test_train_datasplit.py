# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:36:52 2020

@author: Panagiotis Gkanos
"""

import os
os.environ['KERAS_BACKEND']='theano'
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image

##########################################################################################################################

#Function that splits data in train test and validation for each class
def datasplit(data_path,store_path,myclass,batch):
    #print(data_path,store_path,myclass)
    
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

##########################################################################################################################




################################################### MAIN PROGRAMM ###################################################
img_scale='200X' #image scale 
data_path='C:/Users/Panagiotis Gkanos/Desktop/multiclass dataset/'+img_scale+'/classes' #image dataset directory
ben_class=['adenosis','fibroadenoma','phyllodes_tumor','tubular_adenoma'] #benign tumors classes
mal_class=['ductal_carcinoma','lobular_carcinoma','mucinous_carcinoma','papillary_carcinoma'] #malignant tumors classes
batch_ben=[113,260,121,150] #size of each benignal class
batch_mal=[903,170,222,142] #size of each malignant class




####### BENIGNAL DIRECTORIES #######
ben_store_path=[] #list for train-test-valid directories of each benignal class
ben_tumors=['ad_','fib_','pt_','ta_'] #final images' names for each benignal class 
count=0 #counter to determine image name
for i in ben_class:
    ben_store_path.append('C:/Users/Panagiotis Gkanos/Desktop/multiclass dataset/'+img_scale+'/train/'+i+'/'+ben_tumors[count])
    ben_store_path.append('C:/Users/Panagiotis Gkanos/Desktop/multiclass dataset/'+img_scale+'/valid/'+i+'/'+ben_tumors[count])
    ben_store_path.append('C:/Users/Panagiotis Gkanos/Desktop/multiclass dataset/'+img_scale+'/test/'+i+'/'+ben_tumors[count])
    count=count+1
#print(ben_store_path)



####### MALIGNANT DIRECTORIES #######
mal_store_path=[] #final images' names for each malignant class
mal_tumors=['dc_','lc_','mc_','pc_'] #final images' names for each malignant class
count=0 #counter to determine image name
for i in mal_class:
    mal_store_path.append('C:/Users/Panagiotis Gkanos/Desktop/multiclass dataset/'+img_scale+'/train/'+i+'/'+mal_tumors[count])
    mal_store_path.append('C:/Users/Panagiotis Gkanos/Desktop/multiclass dataset/'+img_scale+'/valid/'+i+'/'+mal_tumors[count])
    mal_store_path.append('C:/Users/Panagiotis Gkanos/Desktop/multiclass dataset/'+img_scale+'/test/'+i+'/'+mal_tumors[count])
    count=count+1    
#print(mal_store_path)



print('BENIGNAL CLASSES')
for i in range(len(batch_ben)):
    print(str(i+1)+'/'+str(len(ben_class))+'\nSpliting images of benignal class '+ben_class[i]+'...')
    datasplit(data_path,ben_store_path[i*3:i*3+3],ben_class[i],batch_ben[i])
    print('Completed!\n\n')


print('==============================================================')
print('\n\nMALIGNANT CLASSES')
for i in range(len(batch_mal)):
    print(str(i+1)+'/'+str(len(mal_class))+'\nSpliting images of malignant class '+mal_class[i]+'...')
    datasplit(data_path,mal_store_path[i*3:i*3+3],mal_class[i],batch_mal[i])
    print('Completed!\n\n')
  




























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
