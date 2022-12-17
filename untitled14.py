#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:28:12 2022

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

I1000=pd.read_csv('/Users/inria/Desktop/Cluster/1000meu10epsi',sep=',')
I_1e6=pd.read_csv('/Users/inria/Desktop/Cluster/1e6meu10epsi',sep=',')
I_1e9=pd.read_csv('/Users/inria/Desktop/Cluster/1e9meu10epsi',sep=',')
I_1e11=pd.read_csv('/Users/inria/Desktop/Cluster/1e11meu10epsi',sep=',')


plt.plot(I1000['accuracy'],'*--')
plt.ylabel("Accuracy [%]")
plt.xlabel('No. of Iterations')