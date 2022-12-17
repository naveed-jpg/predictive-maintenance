#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:33:30 2022

@author: inria
"""
# Function for step size optimisation, called from main program #
import random
import numpy as np
from f_cost import f_cost
import time
def maxalphaoptim(tr,tst,M,lmda,w1,df,alpha,alpha1,g,H,V,meu,P,epsi,Max1,Max2,Max3,Max4,tot,start):
    o=random.sample(range(0,len(M)),30)
    for i in o:
        deg=f_cost(i,tr,tst,lmda,w1,M,df)[0]
        for k in range(0,H.shape[0],1):
          for l in range(0,H.shape[1],1):
              if k==i and l==i:
                  H[k,l]=-deg*2*(np.sum(alpha)-alpha[k])*np.sum(alpha)/(np.sum(alpha)**4)
              if i==k and l!=i:
                  H[k,l]=deg*(np.sum(alpha)**2-2*(np.sum(alpha)-alpha[i])*np.sum(alpha))/(np.sum(alpha)**4)
              if k!=i and l==i:
                  H[k,l]=deg*(-1*(np.sum(alpha)**2)+2*alpha[i]*np.sum(alpha))/(np.sum(alpha)**4)
              if k!=i and l!=i:
                  H[k,l]=2*deg*alpha[i]*np.sum(alpha)/(np.sum(alpha)**4)
        
        for x in range(0,len(alpha)):
          if x==i:
              g[x]=(np.sum(alpha)-alpha[x])/(np.sum(alpha)**2)
          else:
              g[x]=-alpha[i]/(np.sum(alpha)**2)       
        epsi=epsi-2*meu*np.matmul(g,np.matmul(H,V))
        epsi=0.001 if epsi<0 else epsi
        epsi=100 if epsi>100 else epsi
        P.append(epsi) # step size
        alpha=alpha+epsi*g*deg
        print(np.argmax(alpha))
        alpha1.append(alpha)
        Max1.append(np.argmax(alpha)) # Alpha Max
        Max2.append(f_cost(np.argmax(alpha),tr,tst,lmda,w1,M,df)[1]) #accuracy
        Max3.append(f_cost(np.argmax(alpha),tr,tst,lmda,w1,M,df)[0])#Cost function
        Max4.append(f_cost(np.argmax(alpha),tr,tst,lmda,w1,M,df)[2])#battery power
        #np.save('P1.npy",np.array(P))
        V=V+g+epsi*np.matmul(H,V)
        end=time.time()
        tot.append(end-start)
    return alpha,g,H,V