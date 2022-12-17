#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 21:51:55 2022

@author: inria
"""

import random
import numpy as np
alpha=np.array([[0,0],[0,0]])
o=random.sample(range(0,5), 2)

o=np.array([0,0])
for i in o:
    for k in np.arange(0,alpha.shape[0],1):
        for l in np.arange(0,alpha.shape[1],1):
            if k==i and l==i:
                alpha[k,l]=1+alpha[k,l]
                print(alpha)
            if k==i and l!=i:
                alpha[k,l]=2+alpha[k,l]
                print(alpha)
            if k!=i and l==i:
                alpha[k,l]=3+alpha[k,l]
                print(alpha)
            if k!=i and l!=i:
                alpha[k,l]=4+alpha[k,l]
                print(alpha)
