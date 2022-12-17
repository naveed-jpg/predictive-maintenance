#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:47:27 2022

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

I=pd.read_csv('/Users/inria/Desktop/Predictive Maintenance/45005500SDG10epsi',sep=',')
plt.plot(I['accuracy'],'*--')
plt.ylabel("Accuracy [%]")
plt.xlabel('No. of Iterations')
