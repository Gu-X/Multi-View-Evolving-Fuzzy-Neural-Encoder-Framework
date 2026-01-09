# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 21:44:36 2026

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
from Evolving_Fuzzy_Neural_Encoder import EFNE

from Feature_Extraction_Pretrained_Model import feature_extraction_pretrained_model
os.environ["CUDA_VISIBLE_DEVICES"]="0"


feature_size=256
iternum=10
chunksize=4
lr=1

# %%  Feature Extraction using pretrained models
# Feature extraction from WHURS19 imageset using ConvNeXtBase and ConvNeXtSmall models
data_directory='WHURS19'
data_name='WHRS'
model_name='ConvNeXtBase'
feature_extraction_pretrained_model(data_directory,data_name,model_name)
model_name='ConvNeXtSmall'
feature_extraction_pretrained_model(data_directory,data_name,model_name)

# Feature extraction from OPTIMAL31 imageset using ConvNeXtBase and ConvNeXtSmall models
data_directory='OPTIMAL31'
data_name='OPTM'
model_name='ConvNeXtBase'
feature_extraction_pretrained_model(data_directory,data_name,model_name)
model_name='ConvNeXtSmall'
feature_extraction_pretrained_model(data_directory,data_name,model_name)

# %% Training evolving fuzzy neural encoders
# Train a supervised evolving fuzzy neural encoder using feature vectors extracted from WHURS19 imageset by ConvNeXtSmall model
data_name='WHRS'
model_name='ConvNeXtSmall'
EFNE_model_CNS_sup=EFNE(feature_size,iternum,chunksize,lr)
EFNE_model_CNS_sup.training(data_name, model_name, 'supervised')
# Training a self-supervised evolving fuzzy neural encoder using feature vectors extracted from WHURS19 imageset by ConvNeXtSmall model
EFNE_model_CNS_selfsup=EFNE(feature_size,iternum,chunksize,lr)
EFNE_model_CNS_selfsup.training(data_name, model_name, 'self_supervised')

# %%
# Train a supervised evolving fuzzy neural encoder using feature vectors extracted from WHURS19 imageset by ConvNeXtBase model
data_name='WHRS'
model_name='ConvNeXtBase'
EFNE_model_CNB_sup=EFNE(feature_size,iternum,chunksize,lr)
EFNE_model_CNB_sup.training(data_name, model_name, 'supervised')

# Training a self-supervised evolving fuzzy neural encoder using feature vectors extracted from WHURS19 imageset by ConvNeXtBase model
EFNE_model_CNB_selfsup=EFNE(feature_size,iternum,chunksize,lr)
EFNE_model_CNB_selfsup.training(data_name, model_name, 'self_supervised')

# %%
# Use trained evolving fuzzy neural encoder to compress feature vectors extracted OPTIMAL31 imageset by ConvNeXtBase and ConvNeXtSmall models
data_name='OPTM'
data0,y=EFNE_model_CNS_sup.feature_extraction(data_name)
data1,_=EFNE_model_CNS_selfsup.feature_extraction(data_name)
data2,_=EFNE_model_CNB_sup.feature_extraction(data_name)
data3,_=EFNE_model_CNB_selfsup.feature_extraction(data_name)

data=np.append(data0.copy(),data1.copy(),axis=1)
data=np.append(data.copy(),data2.copy(),axis=1)
data=np.append(data.copy(),data3.copy(),axis=1)

# %%
# Compute classification performance measures
import sklearn.metrics
def performancemeas(y1,ye0): # performance measure for multi-class classification
    acc=sklearn.metrics.accuracy_score(y1,ye0)  # classification accuracy
    bacc=sklearn.metrics.balanced_accuracy_score(y1,ye0) # balanced classification accuracy
    f1=sklearn.metrics.f1_score(y1,ye0,average='weighted') # f1 scores
    mcc=sklearn.metrics.matthews_corrcoef(y1,ye0) # matthews correlation coefficient 
    return acc,bacc,f1,mcc

from sklearn import svm

# %% Train-test splitting
L,W=data.shape
splitratio=0.8
seq=np.random.permutation(L)
L1=int(L*splitratio)
datatra=data[seq[range(0,L1)],:].copy()
ytra=y[seq[range(0,L1)]].reshape(-1,).copy()
datates=data[seq[range(L1,L)],:].copy()
ytes=y[seq[range(L1,L)]].copy()

# Train a SVM as the classifier
lin_clf = svm.LinearSVC(max_iter=1000)
lin_clf.fit(datatra,ytra)
# Test the trained SVM
dec = lin_clf.decision_function(datates)
ye1=np.argmax(dec,axis=1)
acc,bacc,f1,mcc=performancemeas(ytes,ye1)

