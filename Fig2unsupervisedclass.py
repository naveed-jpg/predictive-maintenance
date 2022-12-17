#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:38:46 2022

@author: inria
"""

import numpy as np
np.set_printoptions(suppress=True, precision=4)
import matplotlib.pyplot as plt
import pandas as pd
IK20accuracy=[]
IK20power=[]
for i in range(0,4,1):
    IK20a=pd.read_csv("/Users/inria/Desktop/Predictive Maintenance/constep1/K20semisupervised%i"%i)
    IK20accuracy.append(IK20a['accuracy'])
    IK20power.append(IK20a['Battery Energy'])   
IK20accuracy=np.array(IK20accuracy)
IK20power=np.array(IK20power)
#fig,ax1=plt.subplots()
#ax1.plot(np.mean(Ipt01epsi,axis=0),label='0.01epsi')
#ax1.plot(np.mean(IK20,axis=0),marker='*',linestyle='--',label='D1')
#ax1.plot(np.mean(I100epsi,axis=0),label='100epsi')
#ax1.plot(np.mean(I1000epsi,axis=0),label='1000epsi')
#ax1.plot(np.median(ISDGOPTM,axis=0),label='SDGAdaptive')
#ax1.plot(np.median(ISDGOPTM100epsialpha,axis=0),label='SDGAdaptivealpha')
#ax1.plot(np.median(I100epsialpha,axis=0),label='SDGAalpha')
#ax1.legend(bbox_to_anchor=(1,1), loc='lower right', ncol=2,prop={'size':8})
#ax1.legend(loc='lower right')
#plt.ylabel('Battery Power [\u03bcwatts]'
#plt.ylabel('Accuracy')
#plt.xlabel('No. of Iterations')

fig,ax1=plt.subplots()
ax1.plot(np.mean(IK20accuracy,axis=0),marker='*',linestyle='--',color='green',label='Accuracy')
ax1.legend(loc='center')

ax2=ax1.twinx()
ax2.plot(np.mean(IK20power,axis=0),marker='*',linestyle='--',label='Battery Power',color='red')
ax2.legend(loc='center right')
ax1.set_xlabel('No. of Iterations')
ax1.set_ylabel('Accuracy')
#ax1.yaxis.label.set_color('red')
#ax1.tick_params(axis='y', colors='red')
ax2.tick_params(axis='y',colors='red')
ax1.tick_params(axis='y',colors='green')

ax2.spines['left'].set_color('green')

ax2.spines['right'].set_color('red')

#ax2.spines['left'].set_color('red')

#ax2.yaxis.label.set_color('green')
#ax2.set_ylabel('Battery Power [\u03bcwatts]')
ax2.set_ylabel('Battery Power[\u03bcWatts]')
plt.show()
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
