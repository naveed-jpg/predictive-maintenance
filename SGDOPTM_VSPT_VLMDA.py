#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:50:07 2022

@author: inria
"""

import numpy as np
np.set_printoptions(suppress=True, precision=4)
import pandas as pd
#from scipy.optimize import Bounds
#import test-train split
#import Logistic regression
import random
import matplotlib.pyplot as plt
import time
#********************#
#Import of functions
#********************#
from maxalphaoptim import maxalphaoptim
#*************************#
#Upload data from sensors
#*************************#

H1=pd.read_csv('/Users/inria/Desktop/M/TwinHouse.csv',sep=';',header=None)
# Inputs Temperatures
Ti2 = H1[7];    # temperature in living room at 187 cm (output)
Ti1=H1[6];      # temperature in living room at 125 cm
Ti=H1[5];     #temperature in living room at 67cm
Tk = H1[11];   # kitchen
Td = H1[12];   # doorway
Tcr = H1[8];    # corridor
Tchl = H1[10];  # Children room
Tb=H1[13];    #Bed room
Ta = H1[3];    # attic
Tg = H1[4];    # cellar
Tv = H1[29];   # ventilation supply air 
df=pd.DataFrame([Ti,Ti1,Ti2,Tk,Td,Tcr,Tchl,Tb]).transpose()
df['Tmean']=df.mean(axis=1)
#ADD few anomalies
for i in random.sample(range(4500,5500),20):
    df.iloc[i,:]=random.sample(range(20,32),df.shape[1])

df.columns=['Lvngrm67','Lvngrm125','Lvngrm187','Ktchn','Drwy','crrdr',
            'Chldrn_rm','Bed_rm','Tmean']
#Generate labels for temperatures
df.insert(9,'TEMP',np.arange(0,df.shape[0],1))
for i in np.arange(0,1305,1):
    if all((0<df.iloc[i,:8])&(df.iloc[i,:8]<=31.1))==True:
        df.iloc[i,9]='Normal'
    if any((31.1<df.iloc[i,:8])&(df.iloc[i,:8]<=31.25))==True:
        df.iloc[i,9]='Hot'
    if any(31.25<(df.iloc[i,:8]))==True:
           df.iloc[i,9]='TooHot'
for i in np.arange(1305,2000,1):
    if all((0<df.iloc[i,:8])&(df.iloc[i,:8]<=27))==True:
        df.iloc[i,9]='Normal'
    if any((27<df.iloc[i,:8])&(df.iloc[i,:8]<=28.5))==True:
        df.iloc[i,9]='Hot'
    if any(28.5<(df.iloc[i,:8]))==True:
        df.iloc[i,9]='TooHot'
for i in np.arange(2000,3479,1):
    if all((0<df.iloc[i,:8])&(df.iloc[i,:8]<=25))==True:
        df.iloc[i,9]='Normal'
    elif any((25<df.iloc[i,:8])&(df.iloc[i,:8]<=26))==True:
        df.iloc[i,9]='Hot'
    if any(26<(df.iloc[i,:8]))==True:
        df.iloc[i,9]='TooHot'
for i in np.arange(3479,4325,1):
    if all((0<df.iloc[i,:8])&(df.iloc[i,:8]<=25.9))==True:
        df.iloc[i,9]='Normal'
    elif any((25.9<df.iloc[i,:8])&(df.iloc[i,:8]<=26.2))==True:
        df.iloc[i,9]='Hot'
    if any(26.2<(df.iloc[i,:8]))==True:
        df.iloc[i,9]='TooHot'
for i in np.arange(4325,5905,1):
    if all((0<df.iloc[i,:8])&(df.iloc[i,:8]<=19))==True:
        df.iloc[i,9]='Normal'
    elif any((19<df.iloc[i,:8])&(df.iloc[i,:8]<=21))==True:
        df.iloc[i,9]='Hot'
    if any(21<(df.iloc[i,:8]))==True:
        df.iloc[i,9]='TooHot'

#All possible combinations of number of sensors, compression and sampling
from itertools import chain, combinations,product
s=np.arange(1,df.shape[1]-1,1) # sensors
x=list(chain(*(combinations(s,i) for i in range(1,1+len(s))))) # Possible combinations of sensors
y=np.arange(4,0,-1) # Possible compression of data
A=[x,y]
M=list(product(*A)) # All combinations of all 

#Initiate an alpha
start=time.time()
Max1=[] #argmax
Max2=[] # Accuracy
Max3=[] # Cummunication cost
Max4=[] # Battery Energy
tot=[] # Time
P=[] #intialize for saving  step size

H=np.tile(np.repeat(0.5,len(M)),(len(M),1))
alpha=np.repeat(0.5,len(M))
epsi=10
meu=1
g=np.repeat(0.5,len(M))
V=np.repeat(0.5,len(M))


for l in range(30):
    tr=np.arange(0,800,1)
    tst=np.arange(800,1305,1)
    lmda=100
    w1=10
    alpha,g,H,V=maxalphaoptim(tr, tst, M, lmda, w1, df, alpha, g, H, V, meu, P, epsi, Max1, Max2, Max3, Max4,tot,start)
    lmda=20
    w1=80
    tr=np.arange(1305,1600,1)
    tst=np.arange(1600,2000,1)
    alpha,g,H,V=maxalphaoptim(tr, tst, M, lmda, w1, df, alpha, g, H, V, meu, P, epsi, Max1, Max2, Max3, Max4,tot,start)
    
    tr=np.arange(2000,2800,1)
    tst=np.arange(2800,3479,1)
    lmda=60
    w1=40
    alpha,g,H,V=maxalphaoptim(tr, tst, M, lmda, w1, df, alpha, g, H, V, meu, P, epsi, Max1, Max2, Max3, Max4,tot, start)
    lmda=40
    w1=60
    tr=np.arange(3479,3900,1)
    tst=np.arange(3900,4325,1)
    alpha,g,H,V=maxalphaoptim(tr, tst, M, lmda, w1, df, alpha, g, H, V, meu, P, epsi, Max1, Max2, Max3, Max4,tot, start)
    
    lmda=80
    w1=20
    tr=np.arange(4325,5200,1)
    tst=np.arange(5200,5905,1)
    alpha,g,H,V=maxalphaoptim(tr, tst, M, lmda, w1, df, alpha, g, H, V, meu, P, epsi, Max1, Max2, Max3, Max4,tot, start)

rs=pd.DataFrame(np.array([P,Max1,Max2,Max3,Max4,tot]).transpose(),columns=['step size','argmax','accuracy','cost function','Battery power','time'])
rs.to_csv('/Users/inria/Desktop/Predictive Maintenance/constep/SGDOPTM10epsisansM',index=False)

fig1=plt.figure(num=1)
plt.plot(Max2,'*--')
plt.xlabel('No. of iteraitons')
plt.ylabel('Accuracy [%]')
fig1=plt.figure()

# plot cost function
fig2=plt.figure(num=2)
plt.plot(Max3,'*--')
plt.xlabel('No. of iteraitons')
plt.ylabel('Cost function')

# plot battery energy
fig3=plt.figure(num=3)
plt.plot(Max4,'*--')
plt.xlabel('No. of iteraitons')
plt.ylabel('Battery power [\u03bcW]')

# plot battery energy
fig4=plt.figure(num=3)
plt.plot(P,'*--')
plt.xlabel('No. of iteraitons')
plt.ylabel('Battery power [\u03bcW]')
