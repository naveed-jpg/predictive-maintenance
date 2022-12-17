#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 09:19:56 2022

@author: inria
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:09:05 2022

@author: inria
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:51:30 2022

@author: inria
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 10:00:04 2022

@author: inria
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Code for 
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
import seaborn as sns
import random
H=pd.read_csv('/Users/inria/Desktop/M/TwinHouse.csv',sep=';',header=None)
W=pd.read_csv('/Users/inria/Desktop/M/TwinWeather.csv',sep=';',header=None)
# Inputs Temperatures
Ti2 = H[7];    # temperature in living room at 187 cm (output)
Ti1=H[6];      # temperature in living room at 125 cm
Ti=H[5];     #temperature in living room at 67cm
Tk = H[11];   # kitchen
Td = H[12];   # doorway
Tcr = H[8];    # corridor
Tchl = H[10];  # Children room
Tb=H[13];    #Bed room
Ta = H[3];    # attic
Tg = H[4];    # cellar
Tv = H[29];   # ventilation supply air 
To = W[2];    # outdoor
Qn = W[5];    # Solar radiations from north
Qs = W[7];    # specific global solar vert. South
Qw = W[8];    # specific global solar vert. West
Qi = H[20];   # el. power living
Qk=H[23]+H[24]; #Kithcne power input minus duct losses
Qd=H[25]; #Doorway Heater
QB=H[26]; #Bedroom Heater
df=pd.DataFrame([Ti,Ti1,Ti2,Tk,Td,Tcr,Tchl,Tb]).transpose()
df['Teman']=df.mean(axis=1)
df.columns=['Lvngrm67','Lvngrm125','Lvngrm187','Ktchn','Drwy','crrdr',
            'Chldrn_rm','Bed_rm','Tmean']
df
#Generate labels for temperatures
df.insert(9,'TEMP',np.arange(0,df.shape[0],1))
for i in np.arange(0,df.shape[0],1):
    if any((0<df.iloc[i,:8])&(df.iloc[i,:8]<=26))==True:
        df.iloc[i,9]='Normal'
    if any((26<df.iloc[i,:8])&(df.iloc[i,:8]<30))==True:
        df.iloc[i,9]='Hot'
    if any(30<=(df.iloc[i,:8]))==True:
        df.iloc[i,9]='TooHot'

#All possible combinations of number of sensors, compression and sampling
from itertools import chain, combinations,product
#s=['Lvngrm67','Lvngrm125','Lvngrm187','Ktchn','Drwy','crrdr','Chldrn_rm','Bed_rm']
s=np.arange(1,df.shape[1]-1,1)
x=list(chain(*(combinations(s,i) for i in range(1,1+len(s)))))
y=np.arange(df.shape[0],1000,-24*6*10)
z=np.arange(4,0,-1)
A=[x,y,z]
M=list(product(*A))
f_costbrt=[]

def f_cost(i_samp,lmda):
    i=i_samp
    #lmda=70 #wieght fot the # of sensors reeduction
    w1=40 #weight for the sensor energy
    del1=[1,0.1,0.01,0.001,0.0001]##For digits after decial points
    rang=40-(-10) # Possible temp range
    n_bits=2*np.log2(rang/del1[M[i][2]]).round(0)# Number of bits req.for communication
    S_rate=M[i][1]/(3600*41*24) # sampling rate divided by number of days and hours per
    #day to calcualte hourly sampling rate
    E_batt=3.6e-6*S_rate*n_bits*len(M[i][0])*1e6 # Energy in microwatts
    
    df.iloc[:,:9].round(M[i][2])
    #random.seed(202)
    #random.sample(range(0,len(M)),5)
    #df1=df.sample(M[i][1])
    random.seed(101)
    df1=df.iloc[random.sample(range(0,len(df)),M[i][1]),:]
    X_train,X_test,y_train,y_test=train_test_split(df1.iloc[:,list(M[i][0])],
                                df1.TEMP,test_size=0.4,random_state=0)
    logmodel=LogisticRegression(multi_class='multinomial',solver='lbfgs',
                                max_iter=10000)
    logmodel.fit(X_train,y_train)
    #Predict the new values
    predictions=logmodel.predict(X_test)
    accuracy=metrics.f1_score(y_test,predictions,average=None)[0]
    confus=metrics.confusion_matrix(y_test,predictions)
    #cost function= Probabaility(Error)-weight*delta_reduced # of sensors-weight*delta_compressed
    return -w1*E_batt+lmda*accuracy.round(3),accuracy,E_batt
    #print(f_cost)
#Cahnge step size here:
epsi=1/1000
epsi
#Initiate an alpha
LC=[]
Err=[]
P_Batt
alpha=np.repeat(0.5,len(M))
g=np.zeros(len(alpha))
random.seed()
o=random.sample(range(0,len(M)),1)
lmda=np.arange(10,120,10)
for lmda in np.arange(10,80,10):
    for i in o:
       deg=epsi*f_cost(i,lmda)[0]/len(alpha)
       for x in range(0,len(alpha)):
            if x==i:
                g[x]=(np.sum(alpha)-alpha[x])/(np.sum(alpha)**2)
            else:
                g[x]=-alpha[x]/(np.sum(alpha)**2)       
       alpha=alpha+g*deg
       LC.append(f_cost(np.argmax(alpha), lmda)[0])
       Err.append(1-f_cost(np.argmax(alpha), lmda)[0])
print("The best cummunication strategy is {} with a cost {} and prediction accuracy {} and consfusion matrix{}"
     .format(M[np.argmax(alpha)],f_cost(np.argmax(alpha))[0],f_cost(np.argmax(alpha))[1],f_cost(np.argmax(alpha))[2]))

        






