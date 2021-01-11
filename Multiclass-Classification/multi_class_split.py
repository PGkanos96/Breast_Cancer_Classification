# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:42:06 2020

@author: Panagiotis Gkanos
"""

import os
#os.environ['KERAS_BACKEND']='theano'
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
    
    #BENIGN CLASSES
    adenosis=glob.glob('C:/Users/Panagiotis Gkanos/Desktop/breast/benign/**/adenosis/**/*-'+img_zoom[i]+'-*.*', recursive=True)
    fibroadenoma=glob.glob('C:/Users/Panagiotis Gkanos/Desktop/breast/benign/**/fibroadenoma/**/*-'+img_zoom[i]+'-*.*', recursive=True)
    phyllodes_tumor=glob.glob('C:/Users/Panagiotis Gkanos/Desktop/breast/benign/**/phyllodes_tumor/**/*-'+img_zoom[i]+'-*.*', recursive=True)
    tubular_adenoma=glob.glob('C:/Users/Panagiotis Gkanos/Desktop/breast/benign/**/tubular_adenoma/**/*-'+img_zoom[i]+'-*.*', recursive=True)
    
    #MALIGNANT CLASSES
    ductal_carcinoma=glob.glob('C:/Users/Panagiotis Gkanos/Desktop/breast/malignant/**/ductal_carcinoma/**/*-'+img_zoom[i]+'-*.*', recursive=True)
    lobular_carcinoma=glob.glob('C:/Users/Panagiotis Gkanos/Desktop/breast/malignant/**/lobular_carcinoma/**/*-'+img_zoom[i]+'-*.*', recursive=True)
    mucinous_carcinoma=glob.glob('C:/Users/Panagiotis Gkanos/Desktop/breast/malignant/**/mucinous_carcinoma/**/*-'+img_zoom[i]+'-*.*', recursive=True)
    papillary_carcinoma=glob.glob('C:/Users/Panagiotis Gkanos/Desktop/breast/malignant/**/papillary_carcinoma/**/*-'+img_zoom[i]+'-*.*', recursive=True)    
    
    
    
    
    
    #BENIGN IMAGES TO FINAL DIRECTORIES
    for f in range(len(adenosis)):
        copy2(adenosis[f], 'C:/Users/Panagiotis Gkanos/Desktop/multiclass dataset/'+img_zoom[i]+'X/classes/adenosis')
        
    for f in range(len(fibroadenoma)):
        copy2(fibroadenoma[f], 'C:/Users/Panagiotis Gkanos/Desktop/multiclass dataset/'+img_zoom[i]+'X/classes/fibroadenoma')

    for f in range(len(phyllodes_tumor)):
        copy2(phyllodes_tumor[f], 'C:/Users/Panagiotis Gkanos/Desktop/multiclass dataset/'+img_zoom[i]+'X/classes/phyllodes_tumor')

    for f in range(len(tubular_adenoma)):
        copy2(tubular_adenoma[f], 'C:/Users/Panagiotis Gkanos/Desktop/multiclass dataset/'+img_zoom[i]+'X/classes/tubular_adenoma')



    #MALIGNANT IMAGES TO FINAL DIRECTORIES
    for f in range(len(ductal_carcinoma)):
        copy2(ductal_carcinoma[f], 'C:/Users/Panagiotis Gkanos/Desktop/multiclass dataset/'+img_zoom[i]+'X/classes/ductal_carcinoma')
        
    for f in range(len(lobular_carcinoma)):
        copy2(lobular_carcinoma[f], 'C:/Users/Panagiotis Gkanos/Desktop/multiclass dataset/'+img_zoom[i]+'X/classes/lobular_carcinoma')

    for f in range(len(mucinous_carcinoma)):
        copy2(mucinous_carcinoma[f], 'C:/Users/Panagiotis Gkanos/Desktop/multiclass dataset/'+img_zoom[i]+'X/classes/mucinous_carcinoma')

    for f in range(len(papillary_carcinoma)):
        copy2(papillary_carcinoma[f], 'C:/Users/Panagiotis Gkanos/Desktop/multiclass dataset/'+img_zoom[i]+'X/classes/papillary_carcinoma')

print(adenosis[1])
