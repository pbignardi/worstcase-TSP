import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.linalg import norm
from pyomo.environ import (
    Var,
    SolverFactory,
    ConstraintList,
    ConcreteModel,
    NonNegativeReals,
    Objective,
    RangeSet,
    Reals,
    Constraint,
    minimize,
    log,
    summation
)

n = 25

X = [[np.array((i/n, j/n)) for i in range(n)]for j in range(n)]
#Z = [np.array((0.5, 0.5)), np.array((0.25, 0.5)), np.array((0.4,0.8)),np.array((0.8,0.2)),np.array((0.2,0.2)),np.array((0.8,0.5))]
Z = [np.array((0.5, 0.5)), np.array((0.25, 0.5)), np.array((0.4,0.8)),np.array((0.8,0.2)),np.array((0.2,0.2))]
L = list()
G = list()
t = 0.2


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
        s = 0
        for i in range(1, len(R)+1):
            s += m.nu[i]
        return integrate_onRi(m.nu, R, Z)+m.nu[0]*t+1/len(R)*(s)

    model = ConcreteModel()
    model.I = RangeSet(0, len(R))
    model.nu = Var(model.I, within=NonNegativeReals)

    # Vincoli
    model.nonNegDen = ConstraintList()
    for i in range(len(R)):
        for point in R[i]:
            model.nonNegDen.add(expr = model.nu[0]*norm(point-Z[i])+model.nu[i+1]>=0)

    model.obj = Objective(rule=objfun)

    opt = SolverFactory('ipopt')
    opt.solve(model)
    return model


def plotRi(R, Z):
    plt.axis([0, 1, 0, 1])
    colors = cm.rainbow(np.linspace(0, 1, len(Z)))
    for z_i, R_i,color in zip(Z, R, colors):
        x_Ri = [point[0] for point in R_i]
        y_Ri = [point[1] for point in R_i]
        plt.scatter(x_Ri, y_Ri, color = color)
        plt.scatter(z_i[0], z_i[1],color = 'k')
    plt.show()


def calcGi(R_i,nu_bar_0,nu_bar_1,L,Z):
    sum = 0
    for point in R_i:
        den = 4*(nu_bar_0*minimum_closest(point, L, Z)+nu_bar_1)
        sum += 1/(n**2*den)
    return -sum


def analytic_center(diamR,L,G,n):
    # diamR è il diametro della regione R
    # L è una lista di vettori che contiene tutte le precedenti iterazioni del
    #   vettore lambda
    # g è una lista di vettori che contiene tutte le precedenti istanze del
    #   vettore g
    
    model = ConcreteModel()
    model.n = RangeSet(0,n-1)
    model.L = Var(model.n, within=Reals)
    model.hyperplane = Constraint(expr = sum([model.L[i] for i in model.L]) == 0)
    model.domain = ConstraintList()
    
    for i in model.L:
        model.domain.add(expr = model.L[i] <= diamR)
    for l,g in zip(L,G):
        domain_expr = sum([g[i]*(model.L[i]) for i in model.L])
        model.domain.add(expr = domain_expr >= sum([g[i]*(l[i]) for i in model.L]))
         
    obj_expr = 1
    for l,g in zip(L,G):
        obj_expr *= sum([g[i]*(model.L[i] - l[i]) for i in model.L])
    for i in model.L:
        obj_expr *= (-diamR + model.L[i])
        
    model.obj = Objective(expr = obj_expr,sense=maximize)
    opt = SolverFactory('ipopt')
    opt.solve(model)
    return [model.L[i].value for i in model.L]
        
L.append(analytic_center(np.sqrt(2), L, G,len(Z)))
l = L[-1]
# Calcola nu_bar
m1 = fixedLambdaProblem(l, X, Z, t)
nu_bar_0 = m1.nu0()
nu_bar_1 = m1.nu1()

UB = integrate_LambdaFixed_UB(nu_bar_0, nu_bar_1, l, X, Z)
R = createRi(l, X, Z)

# Calcola nu_tilda
m2 = coefsFonRiProblem(Z, R, t)

nu_tilda_value = [m2.nu[i].value for i in m2.nu]
LB = integrate_onRi(nu_tilda_value, R, Z)
plotRi(R, Z)

# Calcola g_i
g = [calcGi(R_i, nu_bar_0, nu_bar_1, l, Z) for R_i in R]
G.append(g)


