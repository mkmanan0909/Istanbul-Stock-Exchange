# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:27:08 2018

@author: Manan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('istanbul_stock_exchange.csv')

X=dataset.iloc[:,3:].values
Y=dataset.iloc[:,2:3].values

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=20,random_state=0)
regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_test)

from sklearn import metrics
print(metrics.mean_squared_error(Y_test,Y_pred))

