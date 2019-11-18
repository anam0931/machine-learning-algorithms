#data preprocessing
#importing libraries

import numpy as np #mathematical  tools
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
data=pd.read_csv('Data.csv')
X=data.iloc[:,:-1].values
Y=data.iloc[: , 3].values

'''#taking care of missing Data
from sklearn.preprocessing import Imputer #missing Data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)#object
imputer = imputer.fit(X[:, 1:3])            #attribute
X[:, 1:3] = imputer.transform(X[:, 1:3] ) #method

#encoding categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotecoder = OneHotEncoder(categorical_features = [0])
X = onehotecoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)'''

#splitting data in test and training sset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

'''#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)'''











