# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 21:34:07 2026

@author: XwGu
"""

from tensorflow import keras
import os
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from glob import glob
import numpy as np
from scipy.io import savemat
import pandas as pd


def myprint(s):
    with open('modelsummary.txt','a') as f:
        print(s, file=f)

# %%      
def csv_combination(data_name,model_name):
    x = []
    y = []
    for i, fname in enumerate(glob(data_name+"_"+ model_name+ "_" + "*.csv"), start=1):
        data = pd.read_csv(fname, header=None).values
        data1 = data[:, -1] 
        data2 = data[:, :-1] 
        seq1 = np.unique(data1)
        for j in seq1:
            seq2 = np.where(data1 == j)[0]
            x.append(data2[seq2, :].mean(axis=0))
            y.append(i)
    x = np.vstack(x)
    y = np.array(y)
    savemat(data_name+'_'+model_name+".mat", {"x": x, "y": y})
    
def load_pretrained_model(modelname):
# %%
    if modelname == 'ConvNeXtSmall':
        model=keras.applications.ConvNeXtSmall(
            include_top=True,
            include_preprocessing=True,
            input_shape=None,
            weights="imagenet",
            input_tensor=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax")
        feature_size=768
# %%
    if modelname == 'ConvNeXtBase':
        model=keras.applications.ConvNeXtBase(
            include_top=True,
            include_preprocessing=True,
            input_shape=None,
            weights="imagenet",
            input_tensor=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax")
        feature_size=1024
# %%
    if modelname == 'ConvNeXtLarge':

        model=keras.applications.ConvNeXtLarge(
            include_top=True,
            include_preprocessing=True,
            input_shape=None,
            weights="imagenet",
            input_tensor=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax")
        feature_size=1536    
# %%
    if modelname == 'ConvNeXtxLarge':
        model=keras.applications.ConvNeXtXLarge(
            include_top=True,
            input_shape=None,
            weights="imagenet",
            input_tensor=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
            include_preprocessing=True)
        feature_size=2048
# %%
    model.summary(print_fn=myprint)
    layer_names=[layer.name for layer in model.layers]
    layer_name=layer_names[-3]
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model,feature_size

# %%
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def feature_extraction_pretrained_model(data_directory,data_name,model_name):
    intermediate_layer_model,FZ=load_pretrained_model(model_name)
    test_path1=get_immediate_subdirectories(data_directory)
    for ii in range(0,len(test_path1)):
        test_path=data_directory +'/' +test_path1[ii]+'/'
        imagelist=glob(test_path+'*.jpg')
        Features=np.empty([0,FZ+1],dtype=float);
        for jj in range(0,len(imagelist)):
            ima = image.load_img(imagelist[jj],target_size=(248,248))
            ima = img_to_array(ima)
            ##
            output1=np.empty([0,FZ+1],dtype=float);
            ima1=ima[0:224,0:224,:];
            ima1 = ima1.reshape((1, ima1.shape[0], ima1.shape[1], ima1.shape[2]))
            output2=intermediate_layer_model.predict(ima1)
            output2 =np.append(output2,1+jj)
            output1 =np.append(output1,[output2],axis=0)
            
            ima1=ima[0:224,0:224,:];
            ima1 =np.fliplr(ima1)
            ima1 = ima1.reshape((1, ima1.shape[0], ima1.shape[1], ima1.shape[2]))
            output2=intermediate_layer_model.predict(ima1)
            output2 =np.append(output2,1+jj)
            output1 =np.append(output1,[output2],axis=0)
            ##
            ima1=ima[0:224,24:248,:];
            ima1 = ima1.reshape((1, ima1.shape[0], ima1.shape[1], ima1.shape[2]))
            output2 = intermediate_layer_model.predict(ima1)
            output2 =np.append(output2,1+jj)
            output1 =np.append(output1,[output2],axis=0)
            ima1=ima[0:224,24:248,:];
            ima1 =np.fliplr(ima1)
            ima1 = ima1.reshape((1, ima1.shape[0], ima1.shape[1], ima1.shape[2]))
            output2=intermediate_layer_model.predict(ima1)
            output2 =np.append(output2,1+jj)
            output1 =np.append(output1,[output2],axis=0)
            ##
            ima1=ima[12:236,12:236,:];
            ima1 = ima1.reshape((1, ima1.shape[0], ima1.shape[1], ima1.shape[2]))
            output2 = intermediate_layer_model.predict(ima1)
            output2 =np.append(output2,1+jj)
            output1 =np.append(output1,[output2],axis=0)
            ima1=ima[12:236,12:236,:];
            ima1 =np.fliplr(ima1)
            ima1 = ima1.reshape((1, ima1.shape[0], ima1.shape[1], ima1.shape[2]))
            output2=intermediate_layer_model.predict(ima1)
            output2 =np.append(output2,1+jj)
            output1 =np.append(output1,[output2],axis=0)
            ##
            ima1=ima[24:248,0:224,:];
            ima1 = ima1.reshape((1, ima1.shape[0], ima1.shape[1], ima1.shape[2]))
            output2 = intermediate_layer_model.predict(ima1)
            output2 =np.append(output2,1+jj)
            output1 =np.append(output1,[output2],axis=0)
            ima1=ima[24:248,0:224,:];
            ima1 =np.fliplr(ima1)
            ima1 = ima1.reshape((1, ima1.shape[0], ima1.shape[1], ima1.shape[2]))
            output2=intermediate_layer_model.predict(ima1)
            output2 =np.append(output2,1+jj)
            output1 =np.append(output1,[output2],axis=0)
            ##
            ima1=ima[24:248,24:248,:];
            ima1 = ima1.reshape((1, ima1.shape[0], ima1.shape[1], ima1.shape[2]))
            output2 = intermediate_layer_model.predict(ima1)
            output2 =np.append(output2,1+jj)
            output1 =np.append(output1,[output2],axis=0)
            ima1=ima[24:248,24:248,:];
            ima1 =np.fliplr(ima1)
            ima1 = ima1.reshape((1, ima1.shape[0], ima1.shape[1], ima1.shape[2]))
            output2=intermediate_layer_model.predict(ima1)
            output2 =np.append(output2,1+jj)
            output1 =np.append(output1,[output2],axis=0)
            
            Features=np.append(Features,output1,axis=0)
        np.savetxt(data_name+"_"+ model_name+ "_" + test_path1[ii] +".csv", Features, delimiter=",")
    csv_combination(data_name,model_name)