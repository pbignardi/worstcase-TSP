import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import *
from ac import *
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
    maximize,
    summation
)

h = 100

X = [[np.array((i/h, j/h)) for i in range(h)]for j in range(h)]
#Z = [np.array((0.5, 0.5)), np.array((0.25, 0.5)), np.array((0.4,0.8)),np.array((0.8,0.2)),np.array((0.2,0.2)),np.array((0.8,0.5))]
#Z = [np.array((0.5, 0.5)), np.array((0.25, 0.4)), np.array((0.4,0.8)),np.array((0.8,0.2))]
#Z = [np.array((0.4,0.8)),np.array((0.8,0.2))]
#mu1 = [0.4,0.187]
#mu2 = [0.795,0.490]
mu = [0.5,0.5]
sigma = np.array([[0.07,0],[0,0.07]])
Z = list(filter(lambda x: x[0] <= 1 and x[0] >= 0 and x[1]<= 1 and x[1]>=0, [np.random.default_rng().multivariate_normal(mu,sigma) for i in range(15)]))
t = 100


def minimum_closest(x, L, Z):
    out = [norm(p[0]-x)-p[1] for p in zip(Z, L)]
    return min(out)


def argmin_closest(x, L, Z):
    out = [norm(z-x)-l for z,l in zip(Z, L)]
    return np.argmin(out)


def integrate_LambdaFixed_UB(nu0, nu1, L, X, Z):
    sum = 0
    h = len(X)
    for row in X:
        for point in row:
            den = 4*(nu0*minimum_closest(point, L, Z)+nu1)
            sum += 1/(h**2*den)
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


def integrate_onRi(nu, R, Z,h):
    sum = 0
    for i in range(len(R)):
        for point in R[i]:
            den = 4*(nu[0]*norm(point-Z[i])+nu[i])
            sum += 1/(den*h**2)
    return sum


def coefsFonRiProblem(Z, R, t,h):
    def objfun(m):
        s = 0
        for i in range(1, len(R)+1):
            s += m.nu[i]
        return integrate_onRi(m.nu, R, Z,h)+m.nu[0]*t+1/len(R)*(s)

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
    Markers=["x","s", "o", "*"]
    plt.axis([0, 1, 0, 1])
    colors = cm.rainbow(np.linspace(0, 1, len(Z)))
    i = 0
    for z_i, R_i,color in zip(Z, R, colors):
        x_Ri = [point[0] for point in R_i]
        y_Ri = [point[1] for point in R_i]
        plt.scatter(x_Ri, y_Ri, color = color,s = 20*8/len(Z),alpha=0.08)
        plt.scatter(z_i[0], z_i[1],color = 'k',marker='x')
        i += 1
    plt.show()


def calcGi(R_i,nu_bar_0,nu_bar_1,L,Z,h):
    sum = 0
    for point in R_i:
        den = 4*(nu_bar_0*minimum_closest(point, L, Z)+nu_bar_1)
        sum += 1/(h**2*den)
    return -sum


def analytic_center_brutto(A,b,n):
    # diamR è il diametro della regione R
    # L è una lista di vettori che contiene tutte le precedenti iterazioni del
    #   vettore lambda
    # g è una lista di vettori che contiene tutte le precedenti istanze del
    #   vettore g
    
    model = ConcreteModel()
    model.n = RangeSet(0,n-2)
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
  

#%%
#L.append(analytic_center(np.sqrt(2), L, G,len(Z)))
n = len(Z)
A,b = makeConstraints(np.sqrt(2),n)

UB = 1e10
LB = 1e-10
#while UB-LB > 1e-2:
# Calcola nu_bar
#%%

#b = b.reshape(1,-1)[0]
L = analytic_center_brutto(getAtilde(A),b,n)
#L = L.tolist()
L.append(-sum(L));
m1 = fixedLambdaProblem(L, X, Z, t)
nu_bar_0 = m1.nu0()
nu_bar_1 = m1.nu1()

UB = integrate_LambdaFixed_UB(nu_bar_0, nu_bar_1, L, X, Z)
R = createRi(L, X, Z)

# Calcola nu_tilda
m2 = coefsFonRiProblem(Z, R, t,h)

nu_tilda_value = [m2.nu[i].value for i in m2.nu]
LB = integrate_onRi(nu_tilda_value, R, Z,h)
plotRi(R, Z)

# Calcola g_i
g = [calcGi(R_i, nu_bar_0, nu_bar_1, L, Z, h) for R_i in R]


# Analytic center
#b = b.reshape(-1,1)
A,b = Add_constraints(A, b, L, g)
#A,b = Prune_constraints(A, b, L)


