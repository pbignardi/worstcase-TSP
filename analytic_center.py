import numpy as np
from pyomo.environ import (
    Var,
    SolverFactory,
    ConstraintList,
    Constraint,
    ConcreteModel,
    Objective,
    RangeSet,
    Reals,
    minimize,
    log,
    summation
)


class AnalyticCenter:
    def __init__(self, diamR, m):
        self._diamR = diamR
        self.model = ConcreteModel()
        self.model.range = RangeSet(0, m-1)
        self.model.l = Var(self.model.range, within=Reals)

        # Model Constraints
        self.model.Lambda = ConstraintList()
        self.model.diamRConst = ConstraintList()
        for i in self.model.range:
            self.model.diamRConst.add(expr = self.model.l[i] <= diamR)

        sum = 0
        for i in self.model.range:
            sum += self.model.l[i]
        self.model.hyperplane = Constraint(expr=sum == 0)

        # Model Objective
        def objfun(mod):
            expr = 0
            for i in mod.range:
                expr += -log(self._diamR - mod.l[i])
            return expr
        self.model.obj = Objective(rule=objfun, sense=minimize)

        # Model Optimizer
        self._opt = SolverFactory('ipopt')

    def Optimize(self):
        self._opt.solve(self.model)
        self.model.display()

    def updateAnalyticCenter(self, g, L):
        # Add constraint
        constr_expr = np.dot(g,L)-summation(g,self.model.l)
        self.model.Lambda.add(expr = constr_expr <= 0)

        # Add term to Objective Function
        self.model.obj += -log(constr_expr)

        # Optimize
        self.Optimize()

    def getAC(self):
        return [self.model.l[i].value for i in self.model.l]
