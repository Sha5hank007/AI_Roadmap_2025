# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 16:53:14 2025

@author: GANNOJU SHAHSANK
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dt= pd.read_csv(r"C:\Users\GANNOJU SHAHSANK\Downloads\NARESH IT\test.csv\Salary_Data_linear_regression.csv")

x=dt.iloc[:,:-1]
y=dt.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)

y_pred= regressor.predict(x_test)


#COMPARING PRED AND ACTUAL Y
comparison=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(comparison)

#Visualizing test set
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title('Salary v/s Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



#------------------------------------------------------------------------
m_slope=regressor.coef_
print(m_slope)

c_interceept= regressor.intercept_
print(c_interceept)

#PREDICTION 
y_12=(m_slope*12)+c_interceept
print(y_12)


#-------------------------------------------------------------------------------
dt.mean()
dt['Salary'].mean()

dt.median()

dt['Salary'].mode()

dt.var()
dt.std()

#Coefficent of variation(cv)....
from scipy.stats import variation
variation(dt.values) 
variation(dt['Salary'])

#CORRELATION............

dt.corr()
dt['Salary'].corr(df['YearsExperience'])



#SKEWNESS......
dt.skew()
dt.sem()  #THIS GIVES STANDARD ERROR



#Z-SCORE.................
import scipy.stats as stats
dt.apply(stats.zscore)

stats.zscore(dt['Salary'])

#DEGREE OF FREEDOM

a=dt.shape[0]  #NO OF ROWS
b=dt.shape[1]  #colms

degree_of_freedom=a-b
print(degree_of_freedom)



#SUM OF SQUARES REGRESSION (SSR)
y_mean= np.mean(y)
SSR= np.sum((y_pred-y_mean)**2)
print(SSR)


#SUM PF SQUARE ERROR.......

y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

SST=SSR+SSE
print(SSR)


r_square=1-(SSR/SST)
r_square


#===============================================================


bias= regressor.score(x_test, y_test)
print(bias)

variance=regressor.score(x_test, y_test)
print(variance)

from sklearn.metrics import mean_squared_error
train_mse= mean_squared_error(y_train, regressor.predict(x_train))
test_mse=mean_squared_error(y_test, y_pred)


import pickle
#save model to disk
filename='LINEAR_REGRESSION_MODEL_PRACTICE.pkl'

#open a file in write-binary mode and dump model
with open(filename,'wb') as file:
    pickle.dump(regressor, file)
    
print('MODEL HAS BEEN PICKLED AND SAVED')    

import os
os.getcwd()

















