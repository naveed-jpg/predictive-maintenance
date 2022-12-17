#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 16:22:56 2022

@author: inria
"""

import numpy as np
np.set_printoptions(suppress=True, precision=4)
import matplotlib.pyplot as plt
import pandas as pd
Ipt01epsi=[]
I10epsi=[]
I100epsi=[]
I1000epsi=[]
ISDGOPTM=[]
ISDGOPTMalpha=[]
I10epsialpha=[]
ISDGOPTM100epsialpha=[]
I100epsialpha=[]

X='accuracy'
Y='step size'
for i in range(1,29,1):
    #ISDGOPTMalpha1=pd.read_csv("/Users/inria/Desktop/Predictive Maintenance/constep/cnmeupt01epsi1e6meualpha'%i'"%i)
    #ISDGOPTMalpha.append(ISDGOPTMalpha1[X])
    ISDGOPTM100epsialpha1=pd.read_csv("/Users/inria/Desktop/Predictive Maintenance/constep2/cnmeupt10epsi1e6meualpha4'%a'"%i)
    ISDGOPTM100epsialpha.append(ISDGOPTM100epsialpha1[X])
    #I100epsialpha1=pd.read_csv("/Users/inria/Desktop/Predictive Maintenance/constep/SGDVSPTCNLMDA100epsialpha6%i"%i)
    #I100epsialpha.append(I100epsialpha1[X])

    
#ISDGOPTMalpha=np.array(ISDGOPTMalpha)
ISDGOPTM100epsialpha=np.array(ISDGOPTM100epsialpha)#SDGadaptive @alpha 0.508,0.506,0.504 
#0.501,100 step, alpha not smaller than 10
I100epsialpha=np.array(I100epsialpha)#100 epsi, alpha

fig,ax1=plt.subplots()
#ax1.plot(np.median(Ipt01epsi,axis=0),label='0.01epsi')
#ax1.plot(np.median(I10epsi,axis=0),label='10epsi')
#ax1.plot(np.median(I100epsi,axis=0),label='100epsi')
#ax1.plot(np.median(I1000epsi,axis=0),label='1000epsi')
#ax1.plot(np.median(ISDGOPTM,axis=0),label='SDGAdaptive')
ax1.plot(np.mean(ISDGOPTM100epsialpha,axis=0),label='SDGAdaptivealpha')
#ax1.plot(np.mean(I10epsialpha,axis=0),label='SDGAalpha')
ax1.legend(bbox_to_anchor=(1,1), loc='lower right', ncol=2,prop={'size':8})
plt.ylabel(Y)
plt.xlabel('No.of iterations')
X='step size'
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
