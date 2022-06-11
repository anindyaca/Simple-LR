import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle5 as pickle
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
#some plots to see test vs predicted values, and plot 2: error --> test - predicted values = error
c = [i for i in range (1,len(Y_test)+1,1)]
plt.plot(c,Y_test,color='r',linestyle='--')
plt.plot(c,Y_pred,color='b',linestyle='--')
plt.xlabel('Salary')
plt.ylabel('index')
plt.title('Prediction')
plt.show()

# Error : difference between test and predicted values
error = Y_test - Y_pred
c = [i for i in range(1,len(Y_test)+1 ,1)]
plt.plot(c,error, color='green' , linestyle = '-')
plt.xlabel =('index')
plt.ylabel = ('error')
plt.title = ('error value')
plt.show()

# Model Evaluation
#Linear regression models are mainly evaluated by
#1. Residual plots or error plots error --> actual/test - predicted values gives error/residual
#residual plots exposes biasness in a lr model. residual plots show residual/error values on Y axis and
#predicted values on X-axis. Plot should be randomly  plotted --> means errors across predicted values should be randomly distributed
#if signs of systematic pattern in  distribution, then model is biased

#2. R2 r- squared value --> r2 high ~ 1 -- model represents variance of the dependent variable
# r2 low -- model does not represent the variance of the dependent variable and regression is almost equal to taking the mean value, i.e no information is used from other variables
# r2 < 0 -- r squared negative -- worse than mean value, has negative value when the predictors do not explain the dependent variables at all
# it is not possible to see perfect model with r2 = 1 --> this basically means all values fall on the regression line

# more -- https://towardsdatascience.com/evaluation-metrics-model-selection-in-linear-regression-73c7573208be#:~:text=Evaluation%20metrics%20for%20a%20linear,R%2Dsquared%2C%20and%20RMSE.

# calculating these metrics
from sklearn.metrics import r2_score, mean_squared_error
rmse = mean_squared_error(Y_test , Y_pred , squared= True)
rsq = r2_score(Y_test , Y_pred)
print('root mean squared error = ',rmse)
print('r squared = ',rsq)
# intercept and coefficient of line
print('Intercept of line = ' , lr.intercept_)
print('coefficient of line = ',lr.coef_)
# y = 25202.88 + 9731.20x


#Deploying model to flask
#saving model to disk
pickle.dump(lr, open('lr_model.pkl' , 'wb')) # --> this pkl file is needed to deploy using Flask framework
#test this pkl file model
#model = pickle.load(open('lr_model.pkl' , 'rb'))