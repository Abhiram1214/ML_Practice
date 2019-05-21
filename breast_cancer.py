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

cm = confusion_matrix(y_test, y_predict)
#all values are 1..something wrong..need to scale data

sns.heatmap(cm, annot=True)

#imporving the model ----> normalization and tune svc.

min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train-min_train)/range_train

#not scaled
sns.scatterplot(x=X_train['mean area'], y=X_train['mean smoothness'], hue=y_train)

#scaled
sns.scatterplot(x=X_train_scaled['mean area'], y=X_train_scaled['mean smoothness'], hue=y_train)


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test-min_test)/range_test

svc.fit(X_train_scaled, y_train)

y_predict = svc.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict))
# accuracy = 96%


param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1,0.1,0.01,0.001], 'kernel':['rbf']}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)
grid.fit(X_train_scaled, y_train)


grid.best_params_
#c = 10, gamma=1

grid_predictions = grid.predict(X_test_scaled)

cm_grid = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm_grid, annot=True)

print(classification_report(y_test, grid_predictions))
#97% accuracy












