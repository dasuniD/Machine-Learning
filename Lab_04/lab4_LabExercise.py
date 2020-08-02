import matplotlib #importing Matplotlib module
import matplotlib.pyplot as plt #pyplot is a collection of command style functions
from mpl_toolkits import mplot3d #importing modules for 3D plotting
import numpy as np
import pandas as pd
from sklearn import datasets   #import standard data sets
from sklearn.decomposition import PCA

wine_dataset =datasets.load_wine() #load ’wine’ data set from standard data sets

x=wine_dataset["data"] #defining features values
y =wine_dataset["target"] #defining target variable values

pca = PCA(n_components=3)   #creating 3 component pca
principalComponents = pca.fit_transform(x)    #apply pca to feature values in wine_dataset
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])  #creating a dataframe
#principle components called principalDf
                        

target2=pd.DataFrame(y, columns=['target'])  #convert y to a dataframe giving the column name 'target' 
finalDf = pd.concat([principalDf, target2], axis=1)  #merging it with the principal components dataframe


finalDf['target']=pd.Categorical(finalDf['target'])
my_color=finalDf['target'].cat.codes   #to change the colours of the data points of the plot according to the value of target


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  #creating a 3d plot figure

ax.set_xlabel('Principal Component 1', fontsize = 12)    #to give axis labels
ax.set_ylabel('Principal Component 2', fontsize = 12)
ax.set_zlabel('Principal Component 3', fontsize = 12)
ax.set_title('PCA with 3 Components', fontsize = 20)           #to give a title

scatter= ax.scatter(finalDf['principal component 1'], finalDf['principal component 2'], finalDf['principal component 3'], c=my_color, cmap="Dark2_r", s=30)
#to create the scatter plot with principle components to axis and a colour to the datapoints according to the class

legend1 = ax.legend(*scatter.legend_elements(),loc="upper right", title="Classes")  #to add a legend
ax.add_artist(legend1)
plt.show()  #to visualize the plot
