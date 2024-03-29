import numpy as np #mathematical  tools
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
data=pd.read_csv('Salary_Data.csv')
X=data.iloc[:,:-1].values
Y=data.iloc[: , 1].values


#splitting data in test and training set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#fittig linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting test set results
y_pred = regressor.predict(X_test)

#visualise training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('salry vs exp training set')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()

#visualise test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('salry vs exp training set')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()
