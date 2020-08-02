from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()   #load the iris dataset

x= iris["data"]    #removing the class attribute and getting only the features values

# Determining the optimum value of k using Elbow method.
wcss = [] #within cluster sum of squares
for i in range(1, 11):
  kmeans= KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Applying K-Means algorithm.
kmeans = KMeans(n_clusters=3, random_state=0) #from Elbow method we identified n_clusters=3
closest_cluster_index = kmeans.fit_predict(x)
cluster_centers=kmeans.cluster_centers_



#part(d) answer -- kmeans.cluster_centers_ gives the coordinates of the centroids/centers the clusters formed by k-means algorithm.


# Visualizing the data points and cluster centers in a 3D plot where first three variables of the dataset are the axes.
fig = plt.figure() #creating a figure
ax = fig.add_subplot(111, projection='3d') #creating 3D subplot

ax.set_xlabel('First Variable', fontsize = 12)    #to give axis labels
ax.set_ylabel('Second Variable', fontsize = 12)
ax.set_zlabel('Third Variable', fontsize = 12)
ax.set_title('Data Points and Cluster Centers', fontsize = 20)           #to give a title

plt.scatter(x[:,0], x[:,1], x[:,2], c='blue')    #getting first three variables of the dataset as the axes.
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='red',marker='*')  #to show cluster centers

plt.show() #to show the plot
