import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("Salary_Data.csv")
# print(df.head())

# print(df.describe()) --> gives basic stat for the dataset
# now this is a linear regression problem, simple line graph equation is
# y = mx + c --> here m is slope of graph(line) and c is the constant, y is output (this o/p is to be predicted using the linear regression)
# so independent variable is X and dependent variable is Y (since this is dependent on mx + c)

# now defining X & Y
X = df['YearsExperience']
#print(X.head())
Y = df['Salary']
#print(Y.head())

X_train , X_test, Y_train , Y_test = train_test_split(X,Y, train_size=0.7 , random_state= 100)


X_train = X_train[:,np.newaxis]
X_test = X_test[:,np.newaxis]

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train , Y_train) # fitting the linear model on train data

Y_pred = lr.predict(X_test) # now using the model to predict on test data