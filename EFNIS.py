# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 09:58:08 2024

@author: XwGu
"""
import numpy
import numpy.matlib
import scipy
import math

class EFNIS:
    def __init__(self,xsize,ysize,lr):
        self.ysize=ysize
        self.xsize=xsize
        self.L=0
        self.lr=lr
        self.threshold1=numpy.exp(-3)
        self.threshold2=0.95
        self.acttype='sig'
    
    def initialisation(self,datain):
        self.center=datain
        self.prototype=datain
        self.local_X=datain**2
        self.local_delta=numpy.zeros((1,self.xsize))
        self.Global_mean=datain
        self.Global_X=datain**2
        self.support=[1]
        self.ModelNumber=1
        self.A=self.covarmatrix('vsrp')
        self.L=1
        self.centerlambda=[1]
        self.LocalDensity=[1]
        self.Global_Delta1=0
        self.Global_Delta=numpy.zeros((1,self.xsize))
    
    def forward(self, data):
        L0,W=data.shape
        if self.L==0:
            indx=1
            datain=data[0,:].reshape(1,-1)
            self.initialisation(datain)
        else:
            indx=0
        for i in range(indx,L0):
            datain=data[i,:].reshape(1,-1)
            self.L=self.L+1
            self.Global_mean=(self.Global_mean*(self.L-1)+datain)/self.L
            self.Global_X=(self.Global_X*(self.L-1)+datain**2)/self.L
            self.Global_Delta=abs(self.Global_X-self.Global_mean**2)
            LocalDensity0,_,_=self.firingstrength(datain,self.Global_Delta,self.local_delta)
            if numpy.max(LocalDensity0)<self.threshold1:
                self.ModelNumber=self.ModelNumber+1;
                self.center=numpy.append(self.center,datain,axis=0)
                self.prototype=numpy.append(self.prototype,datain,axis=0)
                self.local_X=numpy.append(self.local_X,datain**2,axis=0)
                self.local_delta=numpy.append(self.local_delta,numpy.zeros((1,self.xsize)),axis=0)
                self.A=numpy.append(self.A,self.covarmatrix('vsrp'),axis=0)
                self.support.append(1)
            else:
                indx1=numpy.argmax(LocalDensity0)
                self.support[indx1]=self.support[indx1]+1
                self.center[indx1,:]=(self.center[indx1,:].copy()*(self.support[indx1]-1)+datain)/self.support[indx1]
                self.local_X[indx1,:]=(self.local_X[indx1,:].copy()*(self.support[indx1]-1)+datain**2)/self.support[indx1]
                self.local_delta[indx1,:]=abs(self.local_X[indx1,:].copy()-self.center[indx1,:].copy()**2)

        LocalDensity0,_,self.Global_Delta1=self.firingstrength(data,self.Global_Delta,self.local_delta)
        self.LocalDensity,self.centerlambda=self.activingrule(LocalDensity0)
        self.YeL,self.Ye=self.outputgeneration(data,self.centerlambda)
        
    def test(self,data):
        L0,W=data.shape
        Global_mean0=(self.Global_mean*self.L+numpy.sum(data,axis=0))/(self.L+L0)
        Global_X0=(self.Global_X*self.L+numpy.sum(data**2,axis=0))/(self.L+L0)
        Global_Delta0=abs(Global_X0-Global_mean0**2)
        LocalDensity0,_,_=self.firingstrength(data,Global_Delta0,self.local_delta)
        _,centerlambda0=self.activingrule(LocalDensity0)
        YeL,Ye=self.outputgeneration(data,centerlambda0)
        return Ye
              
    def backward(self, data, grad):
        dyeL=self.activfunc(self.YeL,'d')
        C0=numpy.zeros((data.shape[0],data.shape[1],self.ModelNumber))
        C1=C0.copy()
        for j in range(0,self.ModelNumber):
            C00=2*(self.prototype[j,:]-data)/self.Global_Delta1[j]
            C00[numpy.isnan(C00)]=0
            C00[numpy.isinf(C00)]=0
            C0[:,:,j]=C00
            C1[:,:,j]=C0[:,:,j]*self.LocalDensity[:,j].reshape(data.shape[0],1)
        C2=numpy.sum(C1,axis=2)
        for j in range(0,self.ModelNumber):
            C1[:,:,j]=(C0[:,:,j]-C2)*self.LocalDensity[:,j].reshape(data.shape[0],1)
        self.outgrad=numpy.zeros((data.shape[0],data.shape[1]))
        data1=numpy.append(numpy.ones((data.shape[0],1)),data,axis=1)
        for j in range(0,self.ModelNumber):
            temp1=numpy.matmul(grad*dyeL[j,:,:],self.A[j,:,1::])*self.centerlambda[:,j].reshape(data.shape[0],1)
            temp2=numpy.matmul(grad,numpy.matmul(self.YeL[j,:,:].transpose(),C1[:,:,j]))
            self.outgrad=self.outgrad+temp1+temp2
            self.A[j,:,:]=self.A[j,:,:]+self.lr*numpy.matmul((grad*dyeL[j,:,:]).transpose(),data1*self.centerlambda[:,j].reshape(data.shape[0],1))
            
    def firingstrength(self,datain,Global_Delta0,local_delta0):
        Global_Delta1=numpy.sum((Global_Delta0+local_delta0)/2,axis=1)
        dist1=(self.cdist(datain,self.prototype).reshape(self.ModelNumber,datain.shape[0]))**2/Global_Delta1.reshape(self.ModelNumber,1)
        LocalDensity=numpy.exp(-dist1.reshape(datain.shape[0],self.ModelNumber))
        LocalDensity[numpy.isnan(LocalDensity)]=1
        centerlambda=LocalDensity/numpy.sum(LocalDensity,axis=1).reshape(datain.shape[0],1)
        return LocalDensity,centerlambda,Global_Delta1
    
    def activingrule(self,LocalDensity):
        LocalDensity0=numpy.zeros((LocalDensity.shape[0],LocalDensity.shape[1]))
        centerlambda0=LocalDensity0.copy()
        for i in range(0,LocalDensity.shape[0]):
            LocalDensity1=LocalDensity[i,:].copy()
            values=numpy.sum(numpy.triu(numpy.matlib.repmat(numpy.sort(LocalDensity1)[::-1].reshape(self.ModelNumber,1),1,self.ModelNumber), k=0),axis=0)
            seq=numpy.argsort(LocalDensity1)[::-1]
            indx1=numpy.where(values>self.threshold2*values[-1])[0][0]+1
            seq1=seq[0:indx1]
            LocalDensity0[i,seq1]=LocalDensity1[seq1]
        centerlambda0=LocalDensity0/numpy.sum(LocalDensity0,axis=1).reshape(LocalDensity0.shape[0],1)
        return LocalDensity0,centerlambda0
    
    def outputgeneration(self,datain,centerlambda0):
        datain1=numpy.append(numpy.ones((datain.shape[0],1)),datain,axis=1)
        Ye=numpy.zeros((datain.shape[0],self.ysize))
        YeL=self.activfunc(numpy.matmul(datain1.reshape(1,datain.shape[0],self.xsize+1),numpy.transpose(self.A,(0,2,1))),'a')
        #YeL=numpy.zeros((self.ModelNumber,datain.shape[0],self.ysize))
        for i in range(0,self.ModelNumber):
            #YeL[i,:,:]=self.activfunc(numpy.matmul(datain1,self.A[i,:,:].transpose()),'a')
            Ye=Ye+YeL[i,:,:]*centerlambda0[:,i].reshape(-1,1)
        return YeL,Ye
                    
    def activfunc(self,x,mode):
        if self.acttype=='sig':
            if mode=='a':
                return  1/(1+numpy.exp(-x))
            if mode=='d':
                return x*(1-x)
       
    
    def pdist(self,data0):
        temp=scipy.spatial.distance.pdist(data0,'euclidean')
        return temp
    
    def cdist(self,data0,data1):
        temp=scipy.spatial.distance.cdist(data0,data1,'euclidean')
        return temp    
    
    def covarmatrix(self,mode0):
        #vsrp
        if mode0=='vsrp':
            M=numpy.random.random((1,self.ysize,self.xsize+1))
            S=numpy.sqrt(self.xsize+1)*2
            A=(S-1)/S
            B=1/S
            M[numpy.where(M<=B)]=-1
            M[numpy.where(M>=A)]=1
            M[numpy.where((M<A)&(M>B))]=0
            return M*numpy.sqrt(S/2)/numpy.sqrt(self.ysize)

        if mode0=='norm':
            return numpy.round(numpy.random.random((1,self.ysize,self.xsize+1)))/(self.xsize+1)