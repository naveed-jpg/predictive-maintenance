#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:19:45 2022

@author: inria
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn import preprocessing
import scipy.stats as ss
from datetime import datetime as dt
from itertools import chain, combinations,product

H=pd.read_excel('/Users/inria/Desktop/M/Experiment_1/Twin_house_exp1_house_O5_10min_ductwork_correction.xls',header=[0,1])
W=pd.read_csv('/Users/inria/Desktop/M/TwinWeather.csv',sep=';',header=None)
# Inputs
#T=H.iloc[:,1] #time
Ti2 = H.iloc[:,7];    # temperature in living room at 187 cm (output)
Ti1=H.iloc[:,6];      # temperature in living room at 125 cm
Ti=H.iloc[:,5];     #temperature in living room at 67cm
Tk = H.iloc[:,11];   # kitchen
Td = H.iloc[:,12];   # doorway
Tcr = H.iloc[:,8];    # corridor
Tchl = H.iloc[:,10];  # Children room
Tb=H.iloc[:,13];    #Bed room
Ta = H.iloc[:,3];    # attic
Tg = H.iloc[:,4];    # cellar
Tv = H.iloc[:,29];   # ventilation supply air 
To = W[2];    # outdoor
Qn = W[5];    # Solar radiations from north
Qs = W[7];    # specific global solar vert. South
Qw = W[8];    # specific global solar vert. West
Qi = H.iloc[:,20];
Qk=H.iloc[:,23]+H.iloc[:,24]; #Kithcne power input minus duct losses
Qd=H.iloc[:,25]; #Doorway Heater
QB=H.iloc[:,26]; #Bedroom Heater
df=pd.DataFrame([Ti,Ti1,Ti2,Tk,Td,Tcr,Tchl,Tb,Qi]).transpose()

df.columns=['Lvngrm67','Lvngrm125','Lvngrm187','Ktchn','Drwy','crrdr',
            'Chldrn_rm','Bed_rm','Power']
fig,ax=plt.subplots()
ax.plot(df.iloc[:,1],label='Temp')
plt.legend()
ax.set_ylabel('Temp')
ax1=ax.twinx()
ax1.plot(df.iloc[:,8],'--r',label='Power')
ax1.set_ylabel('Power')
plt.legend()
TwinPower=[]
TwinNoPower=[]
for i in np.arange(0,len(df),1):
    if df['Power'][i]>10:
        TwinPower.append(df.iloc[i,:])
        
    else:
        TwinNoPower.append(df.iloc[i,:])
#TwinPower=pd.read_csv('/Users/inria/Desktop/Predictive Maintenance/TwinPower')
#TwinPower.iloc[:,8]=1
#TwinNoPower=pd.read_csv('/Users/inria/Desktop/Predictive Maintenance/TwinNoPower')
#TwinNoPower.iloc[:,8]=0
#TwinPower=preprocessing.normalize(TwinPower)
#TwinNoPower=preprocessing.normalize(TwinNoPower)
TwinPower=pd.DataFrame(TwinPower)
TwinNoPower=pd.DataFrame(TwinNoPower)
X=TwinPower
X_dat=TwinNoPower
alpha=0.05
k=20
gam=0.9

N=round(len(TwinPower)/2)
M=len(TwinPower)-N

#X_N=TwinPower.iloc[0:N,0:8]
#X_M=TwinPower.iloc[N:N+M,0:8]
X_N=TwinPower.iloc[:,0:8].sample(frac=0.5,replace=False)
X_M=TwinPower.iloc[:,0:8].drop(X_N.index)
fig,ax=plt.subplots()


nn = NearestNeighbors(k).fit(X_N)
nn.fit(X_N,X_M)
dist, index = nn.kneighbors(X_M)
D=np.sum(np.power(dist,gam),1)
errorcount=0
for i in np.arange(0,len(X_dat),1):

    dist1,index1=nn.kneighbors(np.array(X_dat.iloc[i,0:8]).reshape(1,-1))
    D2=np.sum(np.power(dist1,gam),1)
    
    
    sumD=0
    for k in np.arange(0,M,1):
        if D2>D[k]:
            sumD=sumD+1
    check=sumD/M
    check2=1-alpha
    if sumD/M>1-alpha:
        error=1
        errorcount=errorcount+1
    else:
        error=0
        pvalue=1-check
av_error=errorcount/len(X_dat)











