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
    
    