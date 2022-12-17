#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:01:59 2022

@author: inria
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def f_cost(i,tr,tst,lmda,w1,M,df):
    
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
