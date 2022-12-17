#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 22:40:40 2022

@author: inria
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima
H=pd.read_excel('/Users/inria/Desktop/M/Experiment_1/Twin_house_exp1_house_O5_10min_ductwork_correction.xls',header=[0,1])
W=pd.read_csv('/Users/inria/Desktop/M/TwinWeather.csv',sep=';',header=None)
Tind = pd.read_csv('/Users/inria/Desktop/M/machine_temperature_system_failure.csv',header=0)
Thm = pd.read_csv('/Users/inria/Desktop/M/ambient_temperature_system_failure.csv', header=0)
#plt.plot(Thm.iloc[:,1])
#H.set_index('Time',inplace=True)
#L=H.resample('H').mean()
H.drop(['Date','Time'],axis=1,inplace=True)
index=pd.date_range('21/08/2013',periods=5905,freq='10T')
H.set_index(index,drop=True,append=False,inplace=True,verify_integrity=True)
L=H.resample('30min').mean()
index1=pd.date_range('12/02/2013',periods=22695,freq="5min")
Tind['datetime']=pd.to_datetime(Tind['timestamp'],infer_datetime_format=True)
Tind['Date']= pd.to_datetime(Tind['datetime']).dt.date
Tind['time']=pd.to_datetime(Tind['datetime']).dt.time
Tind.set_index(index1,drop=True,inplace=True,verify_integrity=False)
Tind=Tind[['timestamp','Date','time','value']]
#Moving average
Tind['rolling']=Tind['value'].rolling(window=100,min_periods=1).mean()
#Cummulative mean
Tind['cummulativemean']=Tind['value'].expanding().mean()
#exponential moving average
Tind['exponentialmean']=Tind['value'].ewm(alpha=0.1,adjust=False).mean()
Tind[['value','rolling','cummulativemean','exponentialmean']].plot(xlim=['2013-12-02 21:15:00','2014-02-19 15:25:0'], figsize=(10,5))
from statsmodels.tsa.stattools import adfuller
def adf_test(series):
    result=adfuller(series)
    print('ADF Statistics: {}'.format(result[0]))
    print('P_Value:  {}'.format(result[1]))
    if result[1]<=0.05:
        print("strong evidence against the null hypothesis,reject the null hypothesis. The Data is statinory")
    else:
        print("Data is not stationary, null hypothesis is accepted")
adf_test(Tind['value'])
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#Tind['first difference']=Tind['value']-Tind['value'].shift(1)
#Tind.dropna(inplace=True)
#acf=plot_acf(Tind['first difference'])
#pacf=plot_pacf(Tind['first difference'])
#adf_test(Tind['first difference'])
#Tind['second difference']=Tind['first difference']-Tind['first difference'].shift(1)
#Tind.dropna(inplace=True)
#acf=plot_acf(Tind['second difference'])
#pacf=plot_pacf(Tind['second difference'])
#adf_test(Tind['second difference'])
#Tind['12th difference']=Tind['second difference']-Tind['second difference'].shift(12*12)
#Tind.dropna(inplace=True)
#acf=plot_acf(Tind['12th difference'])
#pacf=plot_pacf(Tind['12th difference'])
#adf_test(Tind['12th difference'])

from statsmodels.tsa.seasonal import seasonal_decompose
decompose=seasonal_decompose(Tind['value'],period=1,model='additive')
trend=decompose.trend
season=decompose.seasonal
residual=decompose.resid
plt.plot(figsize=(12,8))
plt.subplot(411)
plt.plot(Tind.value,label='Original')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(season,label='seasonality')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(residual,label='Residuals')
plt.legend(loc='upper left')
plt.subplots_adjust(hspace=0.5)
plt.show()

from pmdarima.arima import auto_arima
arima_model=auto_arima(Tind.value,start_p=0,d=1,start_q=0,max_p=5,max_d=3,
                       max_q=5,start_P=0,start_Q=0,max_P=5,D=1,max_Q=5,m=1,
                       seasonal=True,n_fits=10,trace=True)

#train_data=Tind['2013-12-02 21:15:00':'2014-01-29 21:15:00'].value
#test_data=Tind['2014-01-29 21:20:00':'2014-02-19 15:25:00'].value
#model_SARIMAX=SARIMAX(train_data, order=(0,1,1),seasonal_order=(1,1,1,12*12))
#model_SARIMAX_fit=model_SARIMAX.fit()
#residuals=pd.DataFrame(model_SARIMAX_fit.resid)
#model_ARIMA_fit.predict(dynamic=False).plot()
#plt.show()
#fig,ax=plt.subplots(1,2)' 
#ax[0].plot(residuals)
#ax[1].plot(residuals,kind='kde')#model_ARIMA_fit.summary()
#plt.plot(Tind.value)
#plt.set_xticklabels(rotation=45,ha='right')
#plt.xticks(rotation=45)
#Tind.iloc[:,0].drop_duplicates(inplace=True)
#Tind.set_index(pd.to_datetime(Tind.timestamp),drop=True,append=False,inplace=True,verify_integrity=True)
