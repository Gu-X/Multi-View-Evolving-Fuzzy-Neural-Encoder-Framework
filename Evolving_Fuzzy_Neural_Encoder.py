# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 22:19:47 2026

@author: XwGu
"""
import sys
import os
import numpy
from MEFNN import MEFNN
from scipy.io import savemat,loadmat

def normalisation(ddd):
    L,W=ddd.shape
    seq=[]
    for ii in range(0,W):
        mi=numpy.min(ddd[:,ii])
        ma=numpy.max(ddd[:,ii])
        delta=ma-mi
        ddd[:,ii]=(ddd[:,ii]-mi)/delta
        if delta!=0:
            seq.append(ii)
    return ddd[:,seq]


def matdataloadunsplit(dr):
    mat_contents = loadmat(dr)
    L1,W1=mat_contents['x'].shape
    y0=numpy.zeros((L1,1),dtype='int')
    data0=numpy.zeros((L1,W1))
    for i in range(0,L1):
        y0[i,:]=mat_contents['y'][0][i]-1
        data0[i,:]=mat_contents['x'][i].reshape(1,-1)
    return data0,y0


class EFNE:
    def __init__(self,feature_size,iternum,chunksize,lr):
        self.iternum=iternum
        self.chunksize=chunksize
        self.lr=lr
        self.feature_size=feature_size
        
    def training(self,data_name,model_name,mode):
        self.model_name=model_name
        self.mode=mode
        data0,y0=matdataloadunsplit(data_name+'_'+self.model_name+".mat")
        if mode == 'supervised':
            CL=int(max(y0)+1)
            y0=numpy.eye(CL)[y0.reshape(-1)]
            L,W=data0.shape
            model=MEFNN(CL,self.lr)
            model.model_init(W,self.feature_size)
            model.model_train(data0,y0,self.iternum,self.chunksize)
        if mode == 'self_supervised':
            data1=normalisation(data0.copy())
            L,W=data0.shape
            model=MEFNN(W,self.lr)
            model.model_init(W,self.feature_size)
            model.model_train(data0,data1,self.iternum,self.chunksize)
        model.model_save('MEFNN_'+ data_name + '_'+ self.model_name +'_'+self.mode)        
    
    def feature_extraction(self,data_name):
        data0,y0=matdataloadunsplit(data_name+'_'+self.model_name+".mat")
        CL=int(max(y0)+1)
        L,W=data0.shape
        lr=1
        model=MEFNN(CL,self.lr)
        model.model_init(W,self.feature_size)
        model.model_load('MEFNN_'+ data_name + '_'+ self.model_name +'_'+self.mode)
        data=model.model_featureextraction(data0.copy())
        return data,y0
        
     