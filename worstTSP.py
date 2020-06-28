import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from pyomo.environ import (
    Var,
    SolverFactory,
    ConstraintList,
    ConcreteModel,
    NonNegativeReals,
    Objective,
    RangeSet
)

n = 30

# Controllare errori di discretizzazione
X = [[np.array((i/n, j/n)) for i in range(n)]for j in range(n)]
Z = [np.array((0.5, 0.5)), np.array((0.25, 0.5))]

# Da calcolare con Analytic center
L = [-0.1, 0.1]
t = 1


def minimum_closest(x, L, Z):
    out = [norm(p[0]-x)-p[1] for p in zip(Z, L)]
    return min(out)


def argmin_closest(x, L, Z):
    out = [norm(z-x)-l for z,l in zip(Z, L)]
    return np.argmin(out)


def integrate_LambdaFixed_UB(nu0, nu1, L, X, Z):
    sum = 0
    for row in X:
        for point in row:
            den = 4*(nu0*minimum_closest(point, L, Z)+nu1)
            sum += 1/(n**2*den)
    return sum


def fixedLambdaProblem(L, X, Z, t):
    def objfun(m):
        return integrate_LambdaFixed_UB(m.nu0, m.nu1, L, X, Z) + m.nu0*t + m.nu1

    model = ConcreteModel()
    model.nu0 = Var(within=NonNegativeReals)
    model.nu1 = Var(within=NonNegativeReals)

    # Vincoli
    model.denGeqZero = ConstraintList()
    for row in X:
        for point in row:
            print(minimum_closest(point, L, Z))
            model.denGeqZero.add(expr = model.nu0*minimum_closest(point, L, Z)+model.nu1 >= 0)

    model.obj = Objective(rule=objfun)

    opt = SolverFactory('ipopt')
    opt.solve(model)
    model.display()
    return model


def createRi(L, X, Z):
    R = [[] for _ in range(len(Z))]
    for row in X:
        for point in row:
            R[argmin_closest(point, L, Z)].append(point)
    return R


def integrate_onRi(nu, R, Z):
    sum = 0
    for i in range(len(R)):
        for point in R[i]:
            den = 4*(nu[0]*norm(point-Z[i])+nu[i])
            sum += 1/(den*n**2)
    return sum


def coefsFonRiProblem(Z, R, t):
    def objfun(m):
        summation = 0
        for i in range(1, len(R)+1):
            summation += m.nu[i]
        return integrate_onRi(m.nu, R, Z)+m.nu[0]*t+1/len(R)*(summation)

    model = ConcreteModel()
    model.I = RangeSet(0, n)
    model.nu = Var(model.I, within=NonNegativeReals)

    # Vincoli
    model.nonNegDen = ConstraintList()
    for i in range(len(R)):
        for point in R[i]:
            model.nonNegDen.add(expr = model.nu[0]*norm(point-Z[i])+model.nu[i+1]>=0)

    model.obj = Objective(rule=objfun)

    opt = SolverFactory('ipopt')
    opt.solve(model)
    model.display()
    return model


def plotRi(R,Z):
    plt.axis([0,1,0,1])
    for z_i, R_i in zip(Z, R):
        x_Ri = [point[0] for point in R_i]
        y_Ri = [point[1] for point in R_i]
        plt.scatter(x_Ri, y_Ri)
        plt.scatter(z_i[0], z_i[1])
    plt.show()



m1 = fixedLambdaProblem(L, X, Z, t)
UB = integrate_LambdaFixed_UB(m1.nu0, m1.nu1, L, X, Z)
R = createRi(L, X, Z)
m2 = coefsFonRiProblem(Z, R, t)
LB = integrate_onRi(m2.nu, R, Z)
plotRi(R, Z)
