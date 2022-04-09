# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 14:30:55 2022

@author: Muthuraj.Jayaseelan
"""
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv(r'C:\Users\muthuraj.jayaseelan\salary_data.csv')
x = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,1].values



#split data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.30,random_state=0)


#building a model
from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(x_train,y_train)


#prediction
y_predict = regress.predict(x_test)


#Visualization
plt.title("Simple Linear Regression")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regress.predict(x_train))
plt.show()



#Visualization with test data
plt.title("Simple Linear Regression with Test Data")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, regress.predict(x_test))
plt.show()


#Visualization with test data and train slope line
plt.title("Simple Linear Regression with Test Data and train slope line")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regress.predict(x_train))
plt.show()


