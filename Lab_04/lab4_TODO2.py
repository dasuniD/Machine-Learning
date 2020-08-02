import matplotlib #importing Matplotlib module
import matplotlib.pyplot as plt #pyplot is a collection of command style functions
from mpl_toolkits import mplot3d #importing modules for 3D plotting
import numpy as np


fig = plt.figure() #creating a figure
ax = fig.add_subplot(111, projection='3d') #creating 3D subplot

xs=([29, 24, 25, 23, 30 ,31, 26, 26, 30, 28])
ys=([ 7, 53 , 33 , 66, 1 ,11, 91, 51, 83, 6])
zs=([-25, -25, -19, -23,-6, -9, -11 , -11,-5, 14])

ax.scatter(xs, ys, zs, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

print(ax.azim)   #to print the azimuth angle

ax.view_init(azim=90, elev=10)  #to change the visualization of 3D plot to a different angle

plt.show()
