#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:38:20 2022

@author: inria
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import scipy.stats as ss
from datetime import datetime as dt

H=pd.read_excel('/Users/inria/Desktop/M/Experiment_1/Twin_house_exp1_house_O5_10min_ductwork_correction.xls',header=[0,1])
W=pd.read_csv('/Users/inria/Desktop/M/TwinWeather.csv',sep=';',header=None)
Tind=pd.read_csv('/Users/inria/Desktop/M/machine_temperature_system_failure.csv',header=0,index_col=0,parse_dates=True)
Thm=pd.read_csv('/Users/inria/Desktop/M/ambient_temperature_system_failure.csv',header=0,index_col=0,parse_dates=True)
# Inputs
#T=H.iloc[:,1] #time
#Ti2 = H.iloc[:,7];    # temperature in living room at 187 cm (output)
#Ti1=H.iloc[:,6];      # temperature in living room at 125 cm
#Ti=H.iloc[:,5];     #temperature in living room at 67cm
#Tk = H.iloc[:,11];   # kitchen
#Td = H.iloc[:,12];   # doorway
#Tcr = H.iloc[:,8];    # corridor
#Tchl = H.iloc[:,10];  # Children room
#Tb=H.iloc[:,13];    #Bed room
#Ta = H.iloc[:,3];    # attic
#Tg = H.iloc[:,4];    # cellar
#Tv = H.iloc[:,29];   # ventilation supply air 
#To = W[2];    # outdoor
#Qn = W[5];    # Solar radiations from north
#Qs = W[7];    # specific global solar vert. South
#Qw = W[8];    # specific global solar vert. West
#Qi = H.iloc[3:,20].str.replace(',','.').astype tsdxc ;p}'(float);   # el. power living
#Qi = H.iloc[:,20];
#Qk=H.iloc[:,23]+H.iloc[:,24]; #Kithcne power input minus duct losses
#Qd=H.iloc[:,25]; #Doorway Heater
#QB=H.iloc[:,26]; #Bedroom Heater
def cusum(data, mean, shift, threshold):
    '''
    Calculate the high and low cumulative sums and use these for anomaly detection. 
    An anomaly is reported if the cumulative sums are beyong a given threshold.
    
    Args: 
        data: (a time series as pandas dataframe; index column is date in datetime format and  
        column 0 is data)
        mean:  mean of the data or other average (float)
        shift: normal shift in the data; standard deviation is recommend (float)
        threshold: threshold to classify point as anomaly (float)

    Returns: 
        cusum: the high and low cumulative sums together with data (pandas dataframe)  
        anomalies: anomalies that above and below threshold (pandas dataframe)
    ''' 
    high_sum = 0.0
    low_sum = 0.0
    anomalies = [] 
    high_sum_final = []
    low_sum_final = []
    index_names = data.index
    data_values = data.values
    for index, item in enumerate(data_values):
        high_sum = max(0, high_sum + item - mean - shift)
        low_sum = max(0, low_sum -item + mean -1.5*shift)
        high_sum_final.append(high_sum)
        low_sum_final.append(low_sum)
        if high_sum > threshold or low_sum < -threshold:
            anomalies.append((index_names[index], item.tolist()))
    cusum = data
    cusum = cusum.assign(High_Cusum=high_sum_final, Low_Cusum=low_sum_final)
    return cusum, anomalies

Tind['datetime']=pd.to_datetime(Tind.index,format = '%Y-%m-%d %H:%M:%S')
Tind['Date']= pd.to_datetime(Tind['datetime'],format='%Y-%m-%d').dt.date
Tind['time']=pd.to_datetime(Tind['datetime'],format='%H:%M:%S').dt.time
Tind.set_index('datetime',drop=True,inplace=True).asfreq('5T')
#Tind['Date']= np.array(Tind['datetime'].dt.date)
#Tind['time']=np.array(Tind['datetime'].dt.time)
#Stattistical Analysis
fig,ax=plt.subplots()
Tind['z_score']=ss.zscore(Tind.value)
ax.plot(Tind.index,Tind['value'], label='sensor reading')
Tind['rolling']=Tind['value'].rolling(window=12*48,min_periods=1).mean()
#Cummulative mean
Tind['cummulativemean']=Tind['value'].expanding().mean()
#exponential moving average
Tind['exponentialmean']=Tind['value'].ewm(com=0.5,adjust=False,min_periods=12).mean()
ax.plot(Tind.index,Tind['rolling'],label='CUM_Mean')
Tind1=pd.DataFrame(Tind.value)
#Tind1.set_index(Tind['datetime'],drop=True,inplace=True)
cusum,anomalies=cusum(Tind1,Tind1.value.mean(),Tind1.value.std(),2)
#plt.plot(np.cumsum(Tind.value))
#ax.plot(Tind['rolling']
ax.axhline(Tind.value[(Tind['z_score']<=-2)].max(),color='r',label='-2 sd',alpha=0.4)
ax.axhline(Tind.value[(Tind['z_score']>=1.5)].max(),color='r',label='+3 sd',alpha=0.4)
plt.legend()
plt.xticks(rotation=45)
ax.axhline(Tind.value.mean(axis=0),color='b',alpha=0.5,label='mean')
# PDF and CDF
fig,ax=plt.subplots(2,1)
ax[0].hist(Tind.value,20,density=True)
ax[0].axvline(Tind.value[(Tind['z_score']<=-2)].max(),color='r',linestyle='--',label='-2 sd',alpha=0.6)
ax[1].hist(Tind.value,30,density=True,histtype='step',cumulative=True)
ax[1].axvline(Tind.value[(Tind['z_score']>=1.6)].max(),color='r',linestyle='--',label='+3 sd',alpha=0.6)

from statsmodels.tsa.seasonal import seasonal_decompose
decompose=seasonal_decompose(Tind['value'],model='additive')

#K-means as unsupervised anomaly detection

#km = KMeans(n_clusters=4).fit(Tind1)

#plt.figure(dpi=120)

#for label in range(4):
#    mask = (km.labels_ == label)
#    plt.plot(Tind1.index[mask], Tind1.iloc[mask, 0], 'o', label=f'Cluster {label}')
#plt.legend();

#Fubction for CUSUM

#X=pd.DataFrame(np.array([To,Qi,Ti]).T)
#nbrs=NearestNeighbors(n_neighbors=4,algorithm='ball_tree').fit(Tind)
#distances,indices=nbrs.kneighbors(Tind)
#L=np.asarray(np.where(distances.mean(axis=1)>1)).flatten()
#plt.axhline()
#plt.plot(distances.mean(axis=1))
#plt.scatter(Qk,Tk)
#plt.scatter(Qk[L],Tk[L],edgecolors='red')
#plt.plot(Ti)
#plt.locator_params(axis="x",nbins=25)
#plt.plot(np.asarray(np.where(distances.mean(axis=1))).flatten())

#fig, ax=plt.subplots()
#ax.plot(Tind)
#ax.plot(Ti1[L],'*')
#fig, ax=plt.subplots()
#ax2=ax.twinx()
#ax3=ax.twinx()
#ax.plot(Ti)
#ax2.plot(Qi)
#ax2.plot(Qi,'r')
