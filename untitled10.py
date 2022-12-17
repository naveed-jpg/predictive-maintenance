#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:03:20 2022

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
from scipy.optimize import Bounds
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
#for i in random.sample(range(4500,5500),20):
#    df.iloc[i,:]=random.sample(range(20,32),df.shape[1])

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
y=np.arange(df.shape[0],144,-24*6*10)
z=np.arange(4,0,-1) # Possible compression of data
A=[x,y,z]
M=list(product(*A)) # All combinations of all 
#f_costbrt=[]
import time
start=time.time()
def f_cost(i_samp):
    i=i_samp
    lmda=100
    #lmda=70 #wieght fot the # of sensors reeduction
    w1=10 #weight for the sensor energy
    del1=[1,0.1,0.01,0.001,0.0001]##For digits after decial points
    rang=40-(-10) # Possible temp range
    n_bits=np.log2(rang/del1[M[i][2]]).round(0)# Number of bits req.for communication
    S_rate=M[i][1]/(3600*41*24) # sampling rate divided by number of days and hours per
    #day to calcualte hourly sampling rate
    P_batt=3.6e-6*S_rate*n_bits*len(M[i][0])*1e6 # Energy in microwatts
    
    df.iloc[:,:9].round(M[i][2])
    #random.seed(202)
    #random.sample(range(0,len(M)),5)
    #df1=df.sample(M[i][1])
    #D=random.uniform(0,1)
    #D1=random.uniform(0,1)
    #df1=df.iloc[:len(M[i][1]),:]
    #X_train,X_test,y_train,y_test=train_test_split(df1.iloc[:,list(M[i][0])],
                                #df1.TEMP,test_size=0.4,random_state=0)
    X_train=df.iloc[4500:5500,list(M[i][0])]
    y_train=df.TEMP.iloc[4500:5500]
    logmodel=LogisticRegression(multi_class='multinomial',solver='lbfgs',
                                max_iter=10000)
    X_test=df.iloc[round(0.8*M[i][1]):round(0.8*M[i][1])+1000,list(M[i][0])]
    y_test=df.TEMP.iloc[round(0.8*M[i][1]):1000+round(0.8*M[i][1])]
    logmodel.fit(X_train,y_train)
    #Predict the new values35
    predictions=logmodel.predict(X_test)
    accuracy=metrics.f1_score(y_test,predictions,average=None)[0]
    #cost function= Probabaility(Error)-weight*delta_reduced # of sensors-weight*delta_compressed
    return -w1*P_batt+lmda*accuracy.round(3),accuracy,P_batt
    #print(f_cost)
#Change step size here:
#Initiate an alpha
Max1=[] #argmax
Max2=[] # Accuracy
Max3=[] # Cummunication cost
Max4=[] # Battery Energy
alpha=np.repeat(0.5,len(M))
#alpha=Z['alpha']
#epsi=Z['epsi']
epsi=10
meu=1e11
#epsi=np.repeat(1/10,len(M))
g=np.repeat(0.5,len(M))
#g=np.zeros(len(alpha))
#g=Z['g']
random.seed()
o=random.sample(range(0,len(M)),100)
#o=[200,400,800,4000]
V=np.repeat(0.5,len(M))
#V=Z['V']

P=[]
for i in o:
    deg=f_cost(i)[0]/len(alpha)
    for x in range(0,len(alpha)):
        #deg=f_cost(i)[0]/len(alpha)
        if x==i:
            g[x]=(np.sum(alpha)-alpha[x])/(np.sum(alpha)**2)
        else:
            g[x]=-alpha[x]/(np.sum(alpha)**2)       
    alpha=alpha+epsi*g*deg
    print(np.argmax(alpha))
    Max2.append(f_cost(np.argmax(alpha))[1])
    Max1.append(np.argmax(alpha))
    Max3.append(f_cost(np.argmax(alpha))[0])
end=time.time()
tot=end-start
tot=np.repeat(tot,len(Max2))
#np.savetxt('accuracySDG10epsi.txt',Max2)
#np.savetxt('argmaxSDG10epsi.txt',Max1)
#np.savetxt('execTimSDG10epsi.txt',tot)
rs=pd.DataFrame(np.array([Max1,Max2,Max3,tot]).transpose(),columns=['argmax','accuracy','communication cost','time'])
rs.to_csv('45005500SDG10epsi',index=False)

