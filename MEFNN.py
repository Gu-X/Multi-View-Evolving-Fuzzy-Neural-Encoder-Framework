# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:41:58 2024

@author: XwGu
"""

from EFNIS import EFNIS
import numpy
import dill

class MEFNN:
    def __init__(self,CL,lr):
        self.CL=CL
        self.lr=lr
          
    def model_init(self,W1,W2):
        self.layer1=EFNIS(W1,W2,self.lr)
        self.layer2=EFNIS(W2,self.CL,self.lr)
    
    def model_train(self,data0,y0,iternum,chunksize):
        L,W=data0.shape
        for i in range(0,iternum):
            seq0=numpy.random.choice(L, L, replace = False)
            data01=data0[seq0,:].copy()
            y01=y0[seq0,:].copy()
            seq1=numpy.append(numpy.arange(0,L,chunksize),L)
            tempG=0
            for j in range(0,len(seq1)-1):
                seq2=range(seq1[j],seq1[j+1])
                self.layer1.forward(data01[seq2,:])
                self.layer2.forward(self.layer1.Ye)
                grad=y01[seq2,:]-self.layer2.Ye
                tempG=tempG+numpy.sum(abs(grad))
                self.layer2.backward(self.layer1.Ye,grad)
                self.layer1.backward(data01[seq2,:],self.layer2.outgrad)
            print(tempG/L) 
            
    def model_save(self,name):
        with open(name+'.pkl', 'wb') as f:
            dill.dump(self.layer1, f)
            dill.dump(self.layer2, f)
        
    def model_load(self,name):
        with open(name+'.pkl', 'rb') as f:
            self.layer1=dill.load(f)
            self.layer2=dill.load(f)
    
    def model_test(self,data1,chunksize):
        L1,W=data1.shape 
        Ye1=numpy.zeros((L1,self.CL))
        seq1=numpy.append(numpy.arange(0,L1,chunksize),L1)
        for j in range(0,len(seq1)-1):
            tempout=self.layer1.test(data1[seq1[j]:seq1[j+1],:])
            Ye1[seq1[j]:seq1[j+1],:]=self.layer2.test(tempout)
        return Ye1
    
    def model_featureextraction(self,data1):
        L1,W=data1.shape 
        Ye1=numpy.zeros((L1,self.layer1.ysize))
        seq1=numpy.append(numpy.arange(0,L1,10),L1)
        for j in range(0,len(seq1)-1):
            Ye1[seq1[j]:seq1[j+1],:]=self.layer1.test(data1[seq1[j]:seq1[j+1],:])
        return Ye1