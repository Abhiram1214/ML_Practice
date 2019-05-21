# -*- coding: utf-8 -*-
"""
Created on Wed May 22 01:56:28 2019

@author: abhiram_ch_v_n_s
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

dataset = pd.read_csv('appdata10.csv')

#EDA
dataset.describe()

#Data Cleaning
dataset['hour'] = dataset['hour'].str.slice(1, 3).astype(int)


#plotting
dataset2 = dataset.copy().drop(columns=['user','screen_list', 'first_open', 'enrolled_date','enrolled'])
dataset2.head()


#Histogram
plt.suptitle("Histograms of Numerical columns", fontsize=10)

for i in range(1, dataset2.shape[1]+1):
    plt.subplot(3,3,i)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i - 1])
    
    vals = np.size(dataset2.iloc[:,i-1].unique())
    
    plt.hist(dataset2.iloc[:,i-1], bins=vals)
    

#Correlation with Response
dataset2.corrwith(dataset.enrolled).plot.bar(figsize = (20, 10),
                  title="correlation with response variables", rot=45, grid=True)    



#correlation matrix
corr = dataset2.corr()
sns.heatmap(corr, annot=True, cmap='YlGnBu')

#Feature Engineering
dataset.dtypes

    #changin to date utility
    
dataset['first_open'] = [parser.parse(rowdata) for rowdata in dataset['first_open']]
dataset['enrolled_date'] = [parser.parse(rowdata) if isinstance(rowdata, str) else rowdata for rowdata in dataset['enrolled_date']]


dataset.dtypes

    #distance in hours
dataset['difference'] = (dataset.enrolled_date - dataset.first_open).astype('timedelta64[h]')

dataset = dataset.drop('differnece', axis=1)
#best hour time to select best cutoff time.

plt.hist(dataset['difference'].dropna())


#this will give us the hours between 0 - 100
plt.hist(dataset['difference'].dropna(), range=[0, 100])

#we can still seee some activity at 20, 30, 40...so we will set the cut off at 48..
dataset.loc[dataset['difference'] > 48, 'enrolled'] = 0

#no use for differnece column
dataset = dataset.drop(['difference', 'enrolled_date', 'first_open'], axis=1)

























