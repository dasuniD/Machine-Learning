import numpy as np #importing numpy module
import matplotlib.pyplot as plt #importing matplotlib modules to plot
import pandas as pd
from sklearn.model_selection import train_test_split #split data set into a train and test set
from numpy.linalg import inv

df = pd.read_csv('Boston_Housing.csv', sep=',', header = None)    #load csv file and remove the header

train_row_count = int((df.shape[0]-1)*(80/100))    #get 80% as the training dataset   

trainset_Y = df[[3]][1:train_row_count+1].values        #Divide the data into feature matrix (x) and response vector (y)
testset_Y = df[[3]][train_row_count+1:].values

trainset_X = df[[0, 1, 2]][1:train_row_count+1].values  
testset_X = df[[0, 1, 2]][train_row_count+1:].values


trainset_X = np.c_[np.ones(train_row_count), trainset_X]        #add a column of ones as the first column of x dataset
testset_X = np.c_[np.ones(df.shape[0] -1 - train_row_count), testset_X]

trainset_Y = trainset_Y.astype(np.float)
testset_Y = testset_Y.astype(np.float)

trainset_X = trainset_X.astype(np.float)
testset_X = testset_X.astype(np.float)

beta_train_set = np.dot(np.dot(np.linalg.inv(np.dot(trainset_X.T, trainset_X)), trainset_X.T), trainset_Y)   #calculate beta

y_train_pred = np.dot(trainset_X, beta_train_set)   #predict values for training set


y_test_pred = np.dot(testset_X, beta_train_set)  #predict values for test set


residual_error_train = trainset_Y - y_train_pred   #calculate the residual errors
residual_error_test = testset_Y - y_test_pred


plt.scatter(y_train_pred, trainset_Y, color = "r", marker = ".", s = 60)     #plot the graphs
plt.scatter(y_test_pred, testset_Y, color = "b", marker = ".", s = 60)
plt.title('Simple Linear Regression')
plt.xlabel("Independent Variable")
plt.ylabel("’Dependent Variable")
plt.show()





