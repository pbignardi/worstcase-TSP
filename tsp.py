from pyomo.environ import *
from utils import CostMatrix
from worstTSP import *
import random
import matplotlib.pyplot as plt
import numpy as np

def tsp(X):
    cost_matrix = CostMatrix(X)
    n = len(X)

    model = ConcreteModel()
    model.N = RangeSet(n)
    model.M = RangeSet(n)
    model.U = RangeSet(2,n)



    model.x = Var(model.N,model.M,within=Binary)
    model.u = Var(model.N,within=NonNegativeIntegers,bounds=(0,n-1))
    model.c = Param(model.N, model.M,initialize=lambda model, i, j: cost_matrix[i-1][j-1])

    ## Constraints
    #   Single visit to each city
    model.const1 = Constraint(model.M,rule = lambda model,M: (sum(model.x[i,M] for i in model.N if i!= M ) == 1))

    # Must depart to a new city each time
    model.constr2 = Constraint(model.N,rule = lambda model,N : (sum(model.x[N,j] for j in model.M if j!=N) == 1))

    # Subtour elimination
    def subtour_elimination(model,i,j):
        if i!=j:
            return model.u[i] - model.u[j] + model.x[i,j] * n <= n-1
        else:
            return model.u[i] - model.u[i] == 0

    model.constr3 = Constraint(model.U,model.N,rule=subtour_elimination)

    def obj_func(model):
        return sum(model.x[i,j] * model.c[i,j] for i in model.N for j in model.M)

    model.obj = Objective(rule=obj_func,sense=minimize)
    opt = SolverFactory('gurobi')
    opt.solve(model)


    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(111)

    return model.x

