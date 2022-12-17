#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:22:01 2022

@author: inria
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:03:38 2022

@author: inria
"""

import numpy as np
np.set_printoptions(suppress=True, precision=4)
import sys
import matplotlib.pyplot as plt
import sklearn.decomposition as decomp
import sklearn.linear_model as linear_model
import sklearn.datasets as sk_data
from sklearn.preprocessing import StandardScaler
import numpy.linalg as nla
import sklearn.svm as svm
import pandas as pd
from scipy.io import loadmat
#from scipy.optimize import Bounds
from pymatreader import read_mat
#import test-train split
from sklearn.model_selection import train_test_split
#import Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
import random
H1=pd.read_csv('/Users/inria/Desktop/M/TwinHouse.csv',sep=';',header=None)
W=pd.read_csv('/Users/inria/Desktop/M/TwinWeather.csv',sep=';',header=None)
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
To = W[2];    # outdoor
Qn = W[5];    # Solar radiations from north
Qs = W[7];    # specific global solar vert. South
Qw = W[8];    # specific global solar vert. West
Qi = H1[20];   # el. power living
Qk=H1[23]+H1[24]; #Kithcne power input minus duct losses
Qd=H1[25]; #Doorway Heater
QB=H1[26]; #Bedroom Heater
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
tot=[] # Time
H=np.tile(np.repeat(0.5,len(M)),(len(M),1))
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
V=np.repeat(0.5,len(M))
#V=Z['V']

P=[]
P.append(epsi)

def maxalpha(tr,tst,M,lmda,w1):
    global V,epsi,alpha,H
    o=random.sample(range(0,len(M)),3)
    for i in o:
        deg=f_cost(i,tr,tst,lmda,w1)[0]/len(alpha)
        for k in range(0,H.shape[0],1):
          for l in range(0,H.shape[1],1):
              if k==i & l==i:
                  H[k,l]=-deg*2*(np.sum(alpha)-alpha[k])*np.sum(alpha)/(np.sum(alpha)**4)
              if i==k & l!=i:
                  H[k,l]=deg*(np.sum(alpha)**2-2*(np.sum(alpha)-alpha[k])*np.sum(alpha))/(np.sum(alpha)**4)
              if k!=i & l==i:
                  H[k,l]=deg*(-1*(np.sum(alpha)**2)+2*alpha[k]*np.sum(alpha))/(np.sum(alpha)**4)
              else:
                  H[k,l]=2*deg*alpha[k]*np.sum(alpha)/(np.sum(alpha)**4)
        
        for x in range(0,len(alpha)):
          if x==i:
              g[x]=(np.sum(alpha)-alpha[x])/(np.sum(alpha)**2)
          else:
              g[x]=-alpha[x]/(np.sum(alpha)**2)       
        epsi=epsi-2*meu*np.matmul(g,np.matmul(H,V))
        epsi=0.001 if epsi<0 else epsi
        P.append(epsi) # step size
        alpha=alpha+epsi*g*deg
        print(np.argmax(alpha))
        Max1.append(np.argmax(alpha)) # Alpha Max
        Max2.append(f_cost(np.argmax(alpha),tr,tst,lmda,w1)[1]) #accuracy
        Max3.append(f_cost(np.argmax(alpha),tr,tst,lmda,w1)[0])#Cost function
        Max4.append(f_cost(np.argmax(alpha),tr,tst,lmda,w1)[2])#battery power
        #np.save('P1.npy",np.array(P))
        V=V+g+epsi*np.matmul(H,V)
        end=time.time()
        tot.append(end-start)

#rs=pd.DataFrame(np.array([P,Max1,Max2,tot]).transpose(),columns=['step size','argmax','accuracy','time'])
#rs.to_csv('1e9meu10epsi',index=False)


tr=np.arange(0,800,1)
tst=np.arange(800,1305,1)
lmda=100
w1=10
maxalpha(tr, tst, M,lmda,w1)
lmda=50
w1=10
tr=np.arange(1305,1600,1)
tst=np.arange(1600,2000,1)
maxalpha(tr, tst, M,lmda,w1)




