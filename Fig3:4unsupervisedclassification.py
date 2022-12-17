#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:56:32 2022

@author: inria
"""

import numpy as np
np.set_printoptions(suppress=True, precision=4)
import matplotlib.pyplot as plt
import pandas as pd
I001200=[]
I10012500=[]
I16002900=[]
X='accuracy'
Y='accuracy'
for i in range(0,8,1):
    #For different data zones
    I001200a=pd.read_csv("/Users/inria/Desktop/Predictive Maintenance/constep/001200semisupervised%i2"%i)
    I001200.append(I001200a[X])
    I10012500a=pd.read_csv("/Users/inria/Desktop/Predictive Maintenance/constep/10012500semisupervised%i2"%i)
    I10012500.append(I10012500a[X])
    I16002900a=pd.read_csv("/Users/inria/Desktop/Predictive Maintenance/constep/16002900semisupervised%i2"%i)
    I16002900.append(I16002900a[X])
    
I001200=np.array(I001200)
I10012500=np.array(I10012500)
I16002900=np.array(I16002900)
fig,ax1=plt.subplots()
#ax1.plot(np.mean(Ipt01epsi,axis=0),label='0.01epsi')
ax1.plot(np.mean(I001200,axis=0),marker='*',linestyle='--',label='D1')
ax1.plot(np.mean(I10012500,axis=0),marker='*',linestyle='--',label='D2')
ax1.plot(np.mean(I16002900,axis=0),marker='*',linestyle='--',label='D3')
#ax1.plot(np.mean(I100epsi,axis=0),label='100epsi')
#ax1.plot(np.mean(I1000epsi,axis=0),label='1000epsi')
#ax1.plot(np.median(ISDGOPTM,axis=0),label='SDGAdaptive')
#ax1.plot(np.median(ISDGOPTM100epsialpha,axis=0),label='SDGAdaptivealpha')
#ax1.plot(np.median(I100epsialpha,axis=0),label='SDGAalpha')
#ax1.legend(bbox_to_anchor=(1,1), loc='lower right', ncol=2,prop={'size':8})
ax1.legend(loc='lower right')
#plt.ylabel('Battery Power [\u03bcwatts]'
plt.ylabel('Accuracy')
plt.xlabel('No. of Iterations')
#Ipt01epsi=[]
#I10epsi=[]
#I100epsi=[]
#I1000epsi=[]
#ISDGOPTM=[]

#for i in range(0,30,1):
#        ISDGOPTM1=pd.read_csv("/Users/inria/Desktop/Predictive Maintenance/constep/SGDOPTMVSPTCNLMDA1e9meu%i"%i)
#        ISDGOPTM.append(ISDGOPTM1[X])
#ISDGOPTM=np.array(ISDGOPTM)
#fig,ax2=plt.subplots()
#ax2.plot(ISDGOPTM1[X],label='SDGAdaptive')
#ax2.legend( loc='lower right', ncol=2,prop={'size':6.6})
#plt.ylabel(X)
#plt.xlabel('No. of iterations')
#plt.legend()
