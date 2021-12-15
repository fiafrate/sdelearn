# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np

from sdelearn import *

import sympy as sym


if __name__ == '__main__':


    # functional mode -------------------------

    def b(x, param):
        out = [0, 0]
        out[0] = param["theta_dr00"] - param["theta_dr01"] * x[0]
        out[1] = param["theta_dr10"] - param["theta_dr11"] * x[1]
        return out



    def A(x, param):
        out = [[0, 0], [0, 0]]
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


    truep = {"theta_dr00": 0, "theta_dr01": -0.5, "theta_dr10": 0, "theta_dr11": -0.5, "theta_di00": 0, "theta_di01": 1,
             "theta_di10": 0, "theta_di11": 1}
    sde.simulate(truep=truep, x0=[1, 2])

    sde.plot()

    # qmle -----------------------------------

    qmle = Qmle(sde)

    # generate some random starting values
    all_param = [p for k in par_names.keys() for p in par_names.get(k)]
    n_param = len(all_param)
    startp = dict(zip(all_param, np.round(np.abs(np.random.randn(n_param)), 1)))

    qmle.fit(startp, method='BFGS')


    qmle.est
    qmle.vcov
    qmle.optim_info


    qmle.predict().plot()




    # symbol mode --------------------

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



    qmle.fit(start=startp, method='L-BFGS-B', bounds = bounds)
    {k1: np.round(np.abs(v1 - v2), 3) for (k1, v1), (k2, v2) in zip(truep.items(), qmle.est.items())}
    qmle.est
    qmle.optim_info



    # lasso ----------------

    # create lasso object. Set a delta > 0 value to use adaptive weights, additionally use
    # the weights argument to apply a specific penalty to each parameter.
    lasso = AdaLasso(sde, qmle, delta=1, start=startp)
    lasso.lambda_max
    lasso.penalty
    lasso.fit()
    # by default no lambda value is chosen and the full path of estimates is computed
    lasso.est_path
    lasso.plot()


    # fit using last 10% obs as validation set (optimal lambda minimizes validation loss)
    lasso.fit(cv=0.1)
    # in this case the estimate corresponding to optimal lambda is computed
    lasso.est
    # estimate of covariance
    lasso.vcov


