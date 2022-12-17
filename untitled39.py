#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:25:04 2022

@author: inria
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:53:01 2022

@author: inria
"""

import numpy as np
np.set_printoptions(suppress=True, precision=4)
import pandas as pd
#from scipy.optimize import Bounds
#import test-train split
#from sklearn.model_selection import train_test_split
#import Logistic regression
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn import preprocessing
#from sklearn.metrics import classification_report
#import seaborn as sns
import random
import matplotlib.pyplot as plt
from itertools import chain, combinations,product

# Load data here
H=pd.read_csv('/Users/inria/Desktop/M/TwinHouse.csv',sep=';',header=None)

# Inputs Temperatures
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
df=pd.DataFrame([Ti,Ti1,Ti2,Tk,Td,Tcr,Tchl,Tb,Qs,Qi]).transpose()

df.columns=['Lvngrm67','Lvngrm125','Lvngrm187','Ktchn','Drwy','crrdr',
            'Chldrn_rm','Bed_rm','solar.E','Power']
fig,ax=plt.subplots()
ax.plot(df.iloc[:,1])
ax1=ax.twinx()
ax1.plot(df.iloc[:,9],'--r')
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
TwinPower=preprocessing.normalize(TwinPower)
TwinNoPower=preprocessing.normalize(TwinNoPower)
TwinPower=pd.DataFrame(TwinPower)
TwinNoPower=pd.DataFrame(TwinNoPower)
X=TwinPower.iloc[:,0:8]
X_dat=TwinNoPower.iloc[:,0:8]
a1=0.05
k=20
gam=0.9
s=np.arange(1,X.shape[1],1)
x=list(chain(*(combinations(s,i) for i in range(1,1+len(s)))))
y=np.arange(X.shape[0],1,-round(len(TwinPower)/4))
z=np.arange(4,0,-1)
A=[x,y,z]
M=list(product(*A))

f_costbrt=[]
import time
start=time.time()

def f_cost(i):
    k=20
    #lmda=10 #wieght fot the # of sensors reeduction
    w1=10 #weight for the sensor energy
    lmda=100
    del1=[1,0.1,0.01,0.001,0.0001]##For digits after decial points
    rang=40-(-10) # Possible temp range
    n_bits=2*np.log2(rang/del1[M[i][2]]).round(0)# Number of bits req.for communication
    S_rate=M[i][1]/(3600*41*24) # sampling rate divided by number of days and hours per
    #day to calcualte hourly sampling rate
    E_batt=3.6e-6*S_rate*n_bits*len(M[i][0])*1e6 # Energy in microwatts
    
    X.round(M[i][2])
    X_dat.round(M[i][2])

    #random.seed(202)
    #random.sample(range(0,len(M)),5)
    #df1=df.sample(M[i][1])
    random.seed(101)
    X1=X.iloc[random.sample(range(0,len(X)),M[i][1]),:]
    #X_dat1=X_dat.iloc[random.sample(range(0,len(X)),M[i][1]),:]
    #X_train,X_test,y_train,y_test=train_test_split(df1.iloc[:,list(M[i][0])],
     #                           df1.TEMP,test_size=0.4,random_state=0)
    X1=X1.iloc[:,list(M[i][0])]
    X_dat1=X_dat.iloc[:,list(M[i][0])]
    N=round(len(X1)/2)
    F=len(X1)-N
    X_N=X1.iloc[0:N,:]
    X_M=X1.iloc[N:N+F,:]
    nn = NearestNeighbors(k).fit(X_N)
    nn.fit(X_N,X_M)
    dist, index = nn.kneighbors(X_M)
    D=np.sum(np.power(dist,gam),1)
    errorcount=0
    for i1 in np.arange(0,len(X_dat1),1):

        dist1,index1=nn.kneighbors(np.array(X_dat1.iloc[i1,:]).reshape(1,-1))
        D2=np.sum(np.power(dist1,gam),1)
        
        
        sumD=0
        for k in np.arange(0,F,1):
            if D2>D[k]:
                sumD=sumD+1
        check=sumD/F
        check2=1-a1
        if sumD/F>1-a1:
            error=1
            errorcount=errorcount+1
        else:
            error=0
            pvalue=1-check
    av_error=errorcount/len(X_dat)
    accuracy=1-av_error

    return -w1*E_batt+lmda*accuracy,av_error,E_batt
    #print(f_cost)
#Cahnge step size here:

epsi=10
epsi

#Initiate an alpha
alpha=np.repeat(0.5,len(M))
g=np.zeros(len(alpha))
random.seed()
#o=random.sample(range(0,len(M)),1)
#lmda=np.arange(10,120,10)
for a in range(1):
    alpha=np.repeat(0.5,len(M))
    g=np.repeat(0.5,len(M))
    o=random.sample(range(0,len(M)),30)
    alpha1=[]
    Max1=[]
    Max2=[]
    Max3=[]
    Max4=[]
    M12=[]
    for i in o:
        deg=epsi*f_cost(i)[0]
        for x in range(0,len(alpha)):
            if x==i:
                g[x]=(np.sum(alpha)-alpha[x])/(np.sum(alpha)**2)
            else:
                g[x]=-alpha[i]/(np.sum(alpha)**2)       
        alpha=alpha+deg*g
        Max1.append(np.argmax(alpha))
        Max2.append(f_cost(np.argmax(alpha))[1])
        Max3.append(f_cost(np.argmax(alpha))[0])
        Max4.append(f_cost(np.argmax(alpha))[2])
        M12.append(M[np.argmax(alpha)])
        alpha1.append(alpha)
    end=time.time()
    tot=end-start
    tot=np.repeat(tot,len(Max2))
    #np.savetxt('accuracySDG10epsi.txt',Max2)
    #np.savetxt('argmaxSDG10epsi.txt',Max1)
    #np.savetxt('execTimSDG10epsi.txt',tot)
rs=pd.DataFrame(np.array([Max1,Max2,Max3,Max4,M12,tot]).transpose(),columns=['argmax','accuracy','communication cost','Battery Energy','communication policy','time'])
    #rs.to_csv('/Users/inria/Desktop/Predictive Maintenance/constep/SDG10epsi%a'%a,index=False)
