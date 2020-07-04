# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:30:49 2020

@author: Paolo
"""
from pyomo.core.base.symbolic import differentiate
from pyomo.core.expr.current import identify_variables, evaluate_expression
import numpy as np


def getHessian(objective):
    gradient = getGradient(objective)
    varList = list(identify_variables(objective.expr))
    hessian = [ differentiate(gradient[i], wrt_list=varList) for i,v in enumerate(varList) ]
    return hessian

def getGradient(objective):
    varList = list(identify_variables(objective.expr))
    gradient = differentiate(objective.expr,wrt_list=varList)
    return gradient

def evalHessian(hessian):
    n = len(hessian)
    H = np.zeros((n,n))
    for i,row in enumerate(hessian):
        for j,expression in enumerate(row):
            H[i][j] = evaluate_expression(expression)
    return H


def makeConstraints(diamR,n):
    b = diamR*np.ones((n,1))
    # Vincoli associati a \lambda_i \leq diamR
    A = np.eye(n)
    return A,b


def Add_constraints(A, b, lambda_bar, g):
    g = np.array(g)
    A = np.vstack([A, -g])
    b = np.vstack([b,-np.dot(g,lambda_bar)])
    return A,b

def Prune_constraints(A,b,optimum):
    n = len(optimum)
    eta = []
    H = np.zeros((n,n))
    for i in range(len(A)):
        H += (b[i]-np.dot(A[i],optimum))**(-2)*(A[i]*A[i].reshape(-1,1))

    for i in range(len(A)):
        eta.append((b[i]-np.dot(A[i],optimum))/(np.sqrt(np.dot(A[i],np.linalg.solve(H,A[i])))))

    # Tolgo i vincoli che hanno eta > m
    to_remove_index = [eta.index(value) for value in eta if value >= len(A)]
    A = np.delete(A, to_remove_index, axis = 0)
    b = np.delete(b, to_remove_index, axis = 0)

    while len(A) >= 4*n:
        index = np.argmax(eta)
        A = np.delete(A, (index), axis = 0)
        b = np.delete(b, (index), axis = 0)
    return A,b


def calculateCostMatrix(X):
    n = len(X)
    c = np.array((n, n))
    for i, x_i in enumerate(X):
        for j, x_j in enumerate(X):
            c[i, j] = np.norm(x_i-x_j)
    return c
