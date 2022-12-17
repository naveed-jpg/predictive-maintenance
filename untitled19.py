#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 13:34:20 2022

@author: inria
"""
import sys
import numpy as np
from numpy import linalg
np.set_printoptions(suppress=True, precision=4)
import pandas as pd
from scipy.optimize import Bounds
#import test-train split
#import Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import random
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
#s=['Lvngrm67','Lvngrm125','Lvngrm187','Ktchn','Drwy','crrdr','Chldrn_rm','Bed_rm']
s=np.arange(1,df.shape[1]-1,1) # sensors
x=list(chain(*(combinations(s,i) for i in range(1,1+len(s))))) # Possible combinations of sensors
#y=np.arange(1000,300,-300) # Sampling rate
y=np.arange(4,0,-1) # Possible compression of data
A=[x,y]
M=list(product(*A)) # All combinations of all 
#f_costbrt=[]

def f_cost(i,tr,tst,lmda,w1):
    
    #lmda=70 #wieght fot the # of sensors reeduction
    #w1=10 #weight for the sensor energy
    del1=[1,0.1,0.01,0.001,0.0001]##For digits after decial points
    rang=40-(-10) # Possible temp range
    n_bits=np.log2(rang/del1[M[i][1]]).round(0)# Number of bits req.for communication
    S_rate=1/(10*60) # sampling rate of 1 sample per 10 minutes
    #day to calcualte hourly sampling rate
    P_batt=3.6e-6*S_rate*n_bits*len(M[i][0])*1e6 # Energy in microwatts
    
    df.iloc[:,:9].round(M[i][1])
    #random.seed(202)
    #random.sample(range(0,len(M)),5)
    #df1=df.sample(M[i][1])
    #D=random.uniform(0,1)
    #D1=random.uniform(0,1)
    #df1=df.iloc[:len(M[i][1]),:]
    #X_train,X_test,y_train,y_test=train_test_split(df1.iloc[:,list(M[i][0])],
                                #df1.TEMP,test_size=0.4,random_state=0)
    X_train=df.iloc[tr,list(M[i][0])]
    y_train=df.TEMP.iloc[tr]
    logmodel=LogisticRegression(multi_class='multinomial',solver='lbfgs',
                                max_iter=10000)
    X_test=df.iloc[tst,list(M[i][0])]
    y_test=df.TEMP.iloc[tst]
    logmodel.fit(X_train,y_train)
    #Predict the new values35
    predictions=logmodel.predict(X_test)
    accuracy=metrics.f1_score(y_test,predictions,average=None)[0]
    #cost function= Probabaility(Error)-weight*delta_reduced # of sensors-weight*delta_compressed
    return -w1*P_batt+lmda*accuracy.round(3),accuracy,P_batt
    #print(f_cost)
#Change step size here:
#Initiate an alpha
import time
start=time.time()
Max1=[] #argmax
Max2=[] # Accuracy
Max3=[] # Cummunication cost
Max4=[] # Battery Energy
Max5=[] #Norm vector
tot=[] # Time
#Z=read_mat('/Users/inria/Desktop/Predictive Maintenance/Z1.mat')
#H=Z['H']
alpha=np.repeat(0.5,len(M))
#alpha=Z['alpha']
#epsi=Z['epsi']
epsi=10
meu=1e9
#epsi=np.repeat(1/10,len(M))
g=np.repeat(0.5,len(M))
#g=np.zeros(len(alpha))
#g=Z['g']

#o=[200,400,800,4000]
#V=Z['V']

P=[]
#P.append(epsi)

def maxalpha(tr,tst,M,lmda,w1):
    global V,epsi,alpha,H
    o=random.sample(range(0,len(M)),30)
    for i in o:
        deg=f_cost(i,tr,tst,lmda,w1)[0]/len(alpha)
        for x in range(0,len(alpha)):
          if x==i:
              g[x]=(np.sum(alpha)-alpha[x])/(np.sum(alpha)**2)
          else:
              g[x]=-alpha[x]/(np.sum(alpha)**2)       

        alpha=alpha+epsi*g*deg
        print(np.argmax(alpha))
        
        Max1.append(np.argmax(alpha)) # Alpha Max
        Max2.append(f_cost(np.argmax(alpha),tr,tst,lmda,w1)[1]) #accuracy
        Max3.append(f_cost(np.argmax(alpha),tr,tst,lmda,w1)[0])#Cost function
        Max4.append(f_cost(np.argmax(alpha),tr,tst,lmda,w1)[2])#battery power
        Max5.append(linalg.norm(g))
        #np.save('P1.npy",np.array(P))
        end=time.time()
        tot.append(end-start)








lmda=40
w1=60
tr=np.arange(3479,3900,1)
tst=np.arange(3900,4325,1)
maxalpha(tr, tst, M,lmda,w1)

lmda=20
w1=80
tr=np.arange(4325,5200,1)
tst=np.arange(5200,5905,1)
maxalpha(tr, tst, M,lmda,w1)

rs=pd.DataFrame(np.array([Max1,Max2,Max3,Max4,Max5,tot]).transpose(),columns=['argmax','accuracy','cost function','Battery power','normvector','time'])
rs.to_csv('/Users/inria/Desktop/Predictive Maintenance/SGD10epsi/SDG10espi9',index=False)


