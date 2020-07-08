# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 23:32:55 2020

@author: Paolo
"""
from worstTSP import solve, argmin_closest, plotRi, createRi
from tsp import tsp
from scipy.optimize import linear_sum_assignment
from utils import CostMatrix, scatterPoints
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm



if __name__ == "__main__":
    
    h = 100
    X = [[np.array((i/h, j/h)) for i in range(h)]for j in range(h)]
    #Z = [np.array((0.5, 0.5)), np.array((0.25, 0.5)), np.array((0.4,0.8)),np.array((0.8,0.2)),np.array((0.2,0.2)),np.array((0.8,0.5))]
    #Z = [np.array((0.5, 0.5)), np.array((0.25, 0.4)), np.array((0.4,0.8)),np.array((0.8,0.2))]
    #Z = [np.array((0.4,0.8)),np.array((0.8,0.2))]
    mu1 = [0.4,0.187]
    mu2 = [0.795,0.490]
    #mu = [0.5,0.5]
    sigma = np.array([[0.07,0],[0,0.07]])
    n = 6
    Z1 = list(filter(lambda x: x[0] <= 1 and x[0] >= 0 and x[1]<= 1 and x[1]>=0, [np.random.default_rng().multivariate_normal(mu1,sigma) for i in range(n)]))
    Z2 = list(filter(lambda x: x[0] <= 1 and x[0] >= 0 and x[1]<= 1 and x[1]>=0, [np.random.default_rng().multivariate_normal(mu2,sigma) for i in range(n)]))
    U = Z1 + Z2
    
    Z1 = list(filter(lambda x: x[0] <= 1 and x[0] >= 0 and x[1]<= 1 and x[1]>=0, [np.random.default_rng().multivariate_normal(mu1,sigma) for i in range(n)]))
    Z2 = list(filter(lambda x: x[0] <= 1 and x[0] >= 0 and x[1]<= 1 and x[1]>=0, [np.random.default_rng().multivariate_normal(mu2,sigma) for i in range(n)]))
    V = Z1 + Z2
    
    while len(U)<len(V):
        V.pop()
    while len(U)>len(V):
        U.pop()
    Z = U + V
    C = CostMatrix(Z)
    C[len(U):,:len(U)] = 1e10*np.ones((len(U),len(U)))
    C[:len(U),len(U):] = 1e10*np.ones((len(U),len(U)))
    C += 1e10*np.eye(len(U)*2)
    row_ind, col_ind = linear_sum_assignment(C)
    t = C[row_ind,col_ind].sum()
    maxIt = 100
    
    # Find worstcase f
    nu_tilda,L = solve(U,X,t,h)
    
    ## Plot region Ri
    R = createRi(L, X, U)
    
    # Assign point to region
    assignement = []
    Z1 = list(filter(lambda x: x[0] <= 1 and x[0] >= 0 and x[1]<= 1 and x[1]>=0, [np.random.default_rng().multivariate_normal(mu1,sigma) for i in range(20)]))
    Z2 = list(filter(lambda x: x[0] <= 1 and x[0] >= 0 and x[1]<= 1 and x[1]>=0, [np.random.default_rng().multivariate_normal(mu2,sigma) for i in range(20)]))
    V = Z1 + Z2
    
    for point in V:
        assignement.append(argmin_closest(point, L, V))
    
    Q = []
    for index in range(len(R)):
        Q.append(list(map(lambda p: p[1],filter(lambda p: p[0]==index,zip(assignement,V)))))
    
    #%%
    ax1 = plt.gca()
    colors = cm.rainbow(np.linspace(0, 1, len(U)))
    
    for z_i, R_i,color in zip(U, R,colors):
        x_Ri = [point[0] for point in R_i]
        y_Ri = [point[1] for point in R_i]
        ax1.scatter(x_Ri, y_Ri, color = color,s = 20*12/len(Z),alpha=0.2)
        ax1.scatter(z_i[0], z_i[1],color = 'k',marker='x')
    
    #%%
    ax1 = plt.gca()
    for cell,c in zip(Q,colors):
        if len(cell)<3:
            continue
        x = tsp(cell)
        for i, x_i in enumerate(cell):
            for j, x_j in enumerate(cell):
                if x[i+1,j+1].value == 1:
                    x_values = [x_i[0], x_j[0]]
                    y_values = [x_i[1], x_j[1]]
                    #ax1.plot(x_values,y_values,color=c)

        for x_i in cell:
            ax1.scatter(x_i[0],x_i[1],marker='o',c='k')
        
    plt.show()
    