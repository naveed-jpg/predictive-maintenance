#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 13:34:20 2022

@author: inria
"""
#********************#
#Import of Packages
#********************#

#import sys
import numpy as np
np.set_printoptions(suppress=True, precision=4)
import pandas as pd
#import test-train split
#import Logistic regression
#from sklearn.linear_model import LogisticRegression
#from sklearn import metrics
import random
import matplotlib.pyplot as plt
import time
#********************#
#Import of functions
#********************#
from maxalpha import maxalpha
#*************************#
#Upload data from sensors
#*************************#

H1=pd.read_csv('/Users/inria/Desktop/M/TwinHouse.csv',sep=';',header=None)
# Variables for sensors data
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
#changing set point temperature 
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
#s=['Lvngrm67','Lvngrm125','Lvngrm187','Ktchn','Drwy','crrdr','Chldrn_rm','Bed_rm']
s=np.arange(1,df.shape[1]-1,1) # sensors
x=list(chain(*(combinations(s,i) for i in range(1,1+len(s))))) # Possible combinations of sensors
#y=np.arange(1000,300,-300) # Sampling rate
y=np.arange(4,0,-1) # Possible compression of data
A=[x,y]
M=list(product(*A)) # All combinations of all 
#f_costbrt=[]

    #print(f_cost)
#Change step size here:
#********************#
#Initiate values
#********************#
start=time.time()
Max1=[] # Intilize for saving argmax
Max2=[] # Intilize for saving Accuracy
Max3=[] # Intilize for saving cost function
Max4=[] # Intilize for saving Battery Energy
Max5=[] # Intilize for saving Norm vector
tot=[] # Intilize for saving Time
M1=[]
#Initialize alpha
alpha=np.repeat(0.5,len(M))
g=np.repeat(0.5,len(M))
#********************#
# Initialize step size
#********************#
epsi=10

for a in range (30):
    start=time.time()
    Max1=[] # Intilize for saving argmax
    Max2=[] # Intilize for saving Accuracy
    Max3=[] # Intilize for saving cost function
    Max4=[] # Intilize for saving Battery Energy
    Max5=[] # Intilize for saving Norm vector
    tot=[] # Intilize for saving Time
    M1=[]
    alpha=np.repeat(0.5,len(M))
    g=np.repeat(0.5,len(M))
    lmda=100
    w1=10
    tr=np.arange(0,800,1) #training data range
    tst=np.arange(800,1305,1) #test data range
    alpha,g=maxalpha(tr, tst, M, lmda, w1, df, alpha, g, epsi, Max1, Max2, Max3, Max4, Max5, M1,tot, start) #Pass values to function
    lmda=20
    w1=80
    tr=np.arange(1305,1600,1) #training data range
    tst=np.arange(1600,2000,1) #test data range
    alpha,g=maxalpha(tr, tst, M, lmda, w1, df, alpha, g, epsi, Max1, Max2, Max3, Max4, Max5, M1,tot, start) #pass values to function
    lmda=60
    w1=40
    tr=np.arange(2000,2800,1) #training data range
    tst=np.arange(2800,3479,1) #test data range
    alpha,g=maxalpha(tr, tst, M, lmda, w1, df, alpha, g, epsi, Max1, Max2, Max3, Max4, Max5, M1,tot, start) #pass values to function
    lmda=40
    w1=60
    tr=np.arange(3479,3900,1) #training data range
    tst=np.arange(3900,4325,1) #test dara range
    alpha,g=maxalpha(tr, tst, M, lmda, w1, df, alpha, g, epsi, Max1, Max2, Max3, Max4, Max5, M1,tot, start) #pass values to function
    lmda=20
    w1=80
    g,alpha
    tr=np.arange(4325,5200,1) #training data range
    tst=np.arange(5200,5905,1) #test data range
    alpha,g=maxalpha(tr, tst, M, lmda, w1, df, alpha, g, epsi, Max1, Max2, Max3, Max4, Max5, M1,tot, start) #pass values to function
    # Save the data  in a single dataframe with  column names
    rs=pd.DataFrame(np.array([Max1,Max2,Max3,Max4,Max5,M1,tot]).transpose(),columns=['argmax','accuracy','cost function','Battery power','normvector','communication policy','time'])
    #Save data to particular file
    rs.to_csv('/Users/inria/Desktop/Predictive Maintenance/constep/SGDVSPTVLMDA10epsi%a'%a,index=False)
# plot accuracy
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









