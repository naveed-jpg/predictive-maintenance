#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:24:20 2022

@author: inria
"""
import random
import numpy as np
from f_cost import f_cost
from numpy import linalg
import time
def maxalpha(tr,tst,M,lmda,w1,df,alpha,g,epsi,Max1,Max2,Max3,Max4,Max5,M1,tot,start):
    o=random.sample(range(0,len(M)),30)
    for i in o:
        deg=f_cost(i,tr,tst,lmda,w1,M,df)[0]
        for x in range(0,len(alpha)):
          if x==i:
              g[x]=(np.sum(alpha)-alpha[x])/(np.sum(alpha)**2)
          else:
              g[x]=-alpha[i]/(np.sum(alpha)**2)       

        alpha=alpha+epsi*deg*g
        print(np.argmax(alpha))
        #alpha1.append(alpha)
        Max1.append(np.argmax(alpha)) # Alpha Max
        Max2.append(f_cost(np.argmax(alpha),tr,tst,lmda,w1,M,df)[1]) #accuracy
        Max3.append(f_cost(np.argmax(alpha),tr,tst,lmda,w1,M,df)[0])#Cost function
        Max4.append(f_cost(np.argmax(alpha),tr,tst,lmda,w1,M,df)[2])#battery power
        Max5.append(linalg.norm(g))
        M1.append(M[np.argmax(alpha)])
        end=time.time()
        tot.append(end-start)
    return alpha,g        #np.save('P1.npy",np.array(P))


