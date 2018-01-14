# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 00:36:01 2018

@author: Utkarsh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:len(dataset.iloc[0,])-1].values
y = dataset.iloc[:,len(dataset.iloc[0,])-1].values

#fiting decision tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x,y)

print(regressor.predict(6.5))

#Visualising the DecisionTreeRegressor
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x,y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.show()