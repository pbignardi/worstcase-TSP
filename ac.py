# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:13:12 2020

@author: Paolo
"""

import numpy as np
from pyomo.environ import *

def analytic_center(A,b):
    # diamR è il diametro della regione R
    # L è una lista di vettori che contiene tutte le precedenti iterazioni del
    #   vettore lambda
    # g è una lista di vettori che contiene tutte le precedenti istanze del
    #   vettore g
    n = A.shape[1]
    model = ConcreteModel()
    model.n = RangeSet(0,n-1)
    model.L = Var(model.n, within=Reals)
    model.domain = ConstraintList()
    
    for a_i, b_i in zip(A,b):
        model.domain.add(expr = sum([a_i[j]*model.L[j] for j in model.L]) <= b_i[0])    
         
    obj_expr = 0
    for a_i, b_i in zip(A,b):
        obj_expr += -log(b_i[0] - sum([a_i[j]*model.L[j] for j in model.L]))
        
    model.obj = Objective(expr = obj_expr,sense=minimize)
    opt = SolverFactory('ipopt')
    opt.solve(model)
    
    return [model.L[i].value for i in model.L]
  

def getAtilde(A):
    last_column = A[:,-1].reshape(-1,1)
    A_tilde = A[:,0:-1]-np.tile(last_column, (1,A.shape[1]-1))
    return A_tilde
