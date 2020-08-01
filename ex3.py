import numpy as np #importing numpy module
import matplotlib.pyplot as plt #importing matplotlib modules to plot
import pandas as pd
from sklearn.model_selection import train_test_split #split data set into a train and test set
from numpy.linalg import inv
import random
from sklearn.linear_model import LinearRegression   #import the linear regression model

dataset = pd.read_csv (r'Boston_Housing.csv')

X= dataset.drop('MEDV', axis=1)
Y= dataset.drop(['RM', 'LSTAT', 'PTRATIO'], axis=1)

#splitting data set into a train and test set with 80% and 20%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

df = pd.DataFrame()

for i in range(49):     #to get 50 samples
    
      x_train_sample = x_train.sample(n=100, random_state=5)     #create random samples of x_train and y_train
      y_train_sample = y_train.sample(n=100, random_state=5)

      log_reg = LinearRegression() #creating an instance of the model
      x= log_reg.fit(x_train_sample,y_train_sample) #fitting the relationship between data
      predictions = log_reg.predict(x_test) #predict labels for test data

      x= pd.DataFrame(predictions)         #create a dataframe from predictions
      df= pd.concat([df, x], axis=1, sort=False)   #keep adding the prediction to the dataframe
      

final_pred = pd.DataFrame()
final_pred['mean'] = df.mean(axis=1)    #get the average of each row of df

residual_error = y_test - final_pred     #residual error

plt.scatter(y_test, final_pred, color = "b", marker = "*", s = 60) #plotting a scatter plot
plt.title('Simple Linear Regression') #adding a title to the graph
plt.xlabel('Independent Variable') #adding axis labels
plt.ylabel('Dependent Variable')
plt.show() #displaying the plot



      



    

