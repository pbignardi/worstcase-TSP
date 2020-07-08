# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 21:28:47 2020

@author: Paolo
"""

import matplotlib.pyplot as plt
from worstTSP import argmin_closest
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create x, y coords
x = np.linspace(0., 1, 100) 
y = np.linspace(0., 1, 100) 
X, Y = np.meshgrid(x, y)
Z = np.zeros(100)
# dummy data
for i in range(len(X)):
    for j in range(len(X[0])):
        x = X[i][j]
        y = Y[i][j]
        point = np.array((x,y))
        k = argmin_closest(point, L, Z)
        Z[i,j] = 1/4*(nu_tilda_value[0]*norm(point-Z[k])+nu_tilda_value[k+1])**(-2)
    

# Create matplotlib Figure and Axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print( X.shape, Y.shape, Z.shape)

# Plot the surface
ax.plot_surface(X, Y, Z)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
