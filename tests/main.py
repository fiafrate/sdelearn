# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
from sympy.combinatorics import cyclic

from sdelearn import *
from sde_data import *
from sde_sampling import *
from sde_model import *
from sde_qmle import*
from sde_lasso import*
import sympy as sym



if __name__ == '__main__':


    # function mode
    def b(x, param):
        out = [0,0]
        out[0]= param["theta_dr00"] - param["theta_dr01"] * x[0]
        out[1] = param["theta_dr10"] - param["theta_dr11"] * x[1]
        return out


    def A(x, param):
        out = [[0,0],[0,0]]
        out[0][0] = param["theta_di00"] + param["theta_di01"] * x[0]
        out[1][1] = param["theta_di10"] + param["theta_di11"] * x[1]
        out[1][0] = 0
        out[0][1] = 0
        return out


    par_names = {"drift": ["theta_dr00", "theta_dr01", "theta_dr10", "theta_dr11"],
                 "diffusion": ["theta_di00", "theta_di01", "theta_di10", "theta_di11"]}
    sde = Sde(sampling=SdeSampling(initial=0, terminal=2, delta=0.01),
              model=SdeModel(b, A, mod_shape=[2, 2], par_names=par_names, mode='fun'))

    print(sde)
    truep = {"theta_dr00": 0, "theta_dr01": -0.5, "theta_dr10": 0, "theta_dr11": -0.5, "theta_di00": -1, "theta_di01": 0, "theta_di10": -1, "theta_di11": 1}
    sde.simulate(truep=truep, x0=[1, 2])

    sde.plot()

    qmle = Qmle(sde)
    startp = dict(zip([p for k in par_names.keys() for p in par_names.get(k)],
                      np.round(np.abs(np.random.randn(len(par_names))), 1)))
    qmle.fit(truep, method='BFGS')
    qmle.est
    qmle.optim_info
    qmle.predict().plot()
    # symbol mode

    n_var = 2
    theta_dr = [sym.symbols('theta_dr{0}{1}'.format(i, j)) for i in range(n_var) for j in range(2)]
    theta_di = [sym.symbols('theta_di{0}{1}'.format(i, j)) for i in range(n_var) for j in range(2)]

    all_param = theta_dr + theta_di
    state_var = [sym.symbols('x{0}'.format(i)) for i in range(n_var)]

    b_expr = np.array([theta_dr[2*i] - theta_dr[2*i+1] * state_var[i] for i in range(n_var)])

    A_expr = np.full((n_var,n_var), sym.sympify('0'))
    np.fill_diagonal(A_expr, [theta_di[2*i] + theta_di[2*i+1] * state_var[i] for i in range(n_var)])

    sde = Sde(sampling=SdeSampling(initial=0, terminal=20, delta=0.01),
              model=SdeModel(b_expr, A_expr, state_var=[s.name for s in state_var]))
    print(sde)

    # truep = {"theta_dr00": 0, "theta_dr01": +0.5, "theta_dr10": 0, "theta_dr11": +0.5, "theta_di00": -1,
    #          "theta_di01": 0, "theta_di10": -1, "theta_di11": 1}
    truep = dict(zip([s.name for s in all_param], np.round(np.abs(np.random.randn(len(all_param))), 1)))
    sde.simulate(truep=truep, x0=np.arange(n_var))

    sde.plot()


    qmle = Qmle(sde)
    startp = dict(zip([s.name for s in all_param], np.round(np.abs(np.random.randn(len(all_param))), 1)))
    box_width = 10
    bounds = [(-0.5*box_width, 0.5*box_width)]*len(all_param) + np.random.rand(len(all_param)*2).reshape(len(all_param), 2)

    qmle.fit(startp, method='AGD', bounds=bounds, cyclic=True, max_it = 1e3)
    {k1: np.round(np.abs(v1-v2), 3) for (k1, v1), (k2, v2) in zip(truep.items(), qmle.est.items())}
    qmle.est
    qmle.optim

    qmle.fit(start=startp, method='L-BFGS-B', bounds = bounds)
    {k1: np.round(np.abs(v1 - v2), 3) for (k1, v1), (k2, v2) in zip(truep.items(), qmle.est.items())}
    qmle.est
    qmle.optim

    qmle.fit(start=startp, method='Newton-CG')
    {k1: np.round(np.abs(v1 - v2), 3) for (k1, v1), (k2, v2) in zip(truep.items(), qmle.est.items())}
    qmle.est

    qmle.gradient(qmle.est)
    qmle.hessian(qmle.est)
    v, a = np.linalg.eigh(qmle.vcov)


    # n-dimesional OU process
    # symbol mode

    n_var = 4
    theta_dr = [sym.symbols('theta_dr{0}{1}'.format(i, j)) for i in range(n_var) for j in range(n_var)]
    theta_di = [sym.symbols('theta_di{0}{1}'.format(i, j)) for i in range(n_var) for j in range(n_var)]

    all_param = theta_dr + theta_di
    state_var = [sym.symbols('x{0}'.format(i)) for i in range(n_var)]

    b_expr = np.array([- np.sum( [theta_dr[i*n_var + j] * state_var[j] for j in range(n_var) ]) for i in range(n_var) ])

    A_expr = np.array([ [ theta_di[i*n_var + j]  for j in range(n_var)] for i in range(n_var)])

    sde = Sde(sampling=SdeSampling(initial=0, terminal=10, delta=0.01),
              model=SdeModel(b_expr, A_expr, state_var=[s.name for s in state_var]))
    print(sde)

    # truep = {"theta_dr00": 0, "theta_dr01": +0.5, "theta_dr10": 0, "theta_dr11": +0.5, "theta_di00": -1,
    #          "theta_di01": 0, "theta_di10": -1, "theta_di11": 1}
    truep = dict(zip([s.name for s in all_param], np.round(np.abs(np.random.randn(len(all_param))), 1)))
    sde.simulate(truep=truep, x0=np.arange(n_var)+1)

    sde.plot()

    qmle = Qmle(sde)
    startp = dict(zip([s.name for s in all_param], 0.1*np.round(np.abs(np.random.randn(len(all_param))), 1)))
    box_width = 10
    bounds = [(-0.5*box_width, 0.5*box_width)]*len(all_param) + np.random.rand(len(all_param)*2).reshape(len(all_param), 2)

    qmle.fit(startp, method='AGD', bounds=bounds)
    {k1: np.round(np.abs(v1 - v2), 3) for (k1, v1), (k2, v2) in zip(truep.items(), qmle.est.items())}
    qmle.est
    qmle.optim

    qmle.fit(start=startp, method='L-BFGS-B', bounds=bounds)
    {k1: np.round(np.abs(v1 - v2), 3) for (k1, v1), (k2, v2) in zip(truep.items(), qmle.est.items())}
    qmle.est
    qmle.optim

    v, a = np.linalg.eigh(qmle.vcov)


    # lasso ----------------

    lasso = AdaLasso(sde, qmle)
