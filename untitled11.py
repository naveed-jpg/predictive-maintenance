#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 08:48:51 2022

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

I=pd.read_csv('/Users/inria/Desktop/Predictive Maintenance/1000meu10epsi',sep=',')
I1=pd.read_csv('/Users/inria/Desktop/Predictive Maintenance/1e9meu10epsi',sep=',')
I2=I1.rename(columns={'accuracy':'argmax','argmax','accuracy'})
I3=pd.read_csv('/Users/inria/Desktop/Predictive Maintenance/SDG10epsi',sep=',')

plt.plot(I3['Battery Energy'],'*--')
plt.xlabel('No. of iterations')
plt.ylabel('Battery Power (\u03bcW)')

