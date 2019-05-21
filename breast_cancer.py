#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 02:35:35 2019

@author: abhiram
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#-----Analysis----------
cancer.feature_names

cancer.keys()

cancer['DESCR']

cancer['data'].shape

#------------


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target']))


df_cancer.isnull().sum()


#Visualization

sns.pairplot(df_cancer,hue='target', vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness'])

sns.countplot(df_cancer['target'])
sns.scatterplot(x='mean area', y='mean smoothness', hue='target', data=df_cancer)

plt.figure(figsize= (20,10))
sns.heatmap(df_cancer.corr(), annot=True)

#Model Training

X = df_cancer.drop('target', axis=1)

y = df_cancer['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

svc = SVC()
svc.fit(X_train, y_train)

#evaluate the model
y_predict = svc.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_predict)







