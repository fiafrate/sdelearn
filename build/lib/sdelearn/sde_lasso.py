import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

from sdelearn import Sde
from sde_qmle import Qmle
from sde_learner import SdeLearner
import warnings

import copy

class AdaLasso(SdeLearner):
    def __init__(self, sde, base_estimator, weights=None, delta=0, penalty=None, **kwargs):
        """

        :param sde: an Sde object
        :param base_estimator: a base estimator of class SdeLearner or a string naming a subclass for constructing one (e.g. Qmle)
        :param weights: adaptive weights/penalties for each parameter, in the form of a dictionary `{param: value}`,
            defaults to None, in that case all weights are set to 1
        :param delta: adjusts adaptive penalties as w_i / est_i^delta, defaults to zero meaning the adaptive weights
        given in argument `weights` remain unchanged
        :param penalty: grid of lambda values at which to evaluate the solution path, defaults to None meaning that
            1000 log-spaced values will be used from 0 to lambda_max
        :**kwargs: arguments to be passed to fit method of base estimator if not already fitted
        """
        super().__init__(sde=sde)

        self.base_est = None
        self.ini_est = None
        self.ini_hess = None

        # instantiate base est if not already present
        if isinstance(base_estimator, SdeLearner):
            self.base_est = base_estimator
        else:
            if base_estimator == 'Qmle':
                self.base_est = Qmle(sde)

        # fit base estimator if not already fitted
        if self.base_est.est is None:
            self.base_est.fit(**kwargs)
            self.ini_est = self.base_est.est
        else:
            self.ini_est = self.base_est.est

        # extract hessian matrix at max point
        self.ini_hess0 = np.linalg.inv(self.base_est.vcov)
        # fix hessian matrix if not symmetric positive definite
        self.ini_hess0 = 0.5 * (self.ini_hess0 + self.ini_hess0.T)
        v, a = np.linalg.eigh(self.ini_hess0)
        # set to zero negative eigs + some small noise
        v[v < 0] = 0.001 * np.abs(np.random.randn(len(v[v < 0])))
        # lipschitz constant of quadratic part
        self.lip = np.max(v)
        self.ini_hess = a @ np.diag(v) @ a.transpose()

        self.delta = delta
        self.weights = weights if weights is not None else dict(zip(sde.model.param, [1] * len(sde.model.param)))

        self.w_ada = np.array(list(self.weights.values())) / \
                     np.power(np.abs(np.array(list(self.ini_est.values()))), delta)

        self.lambda_max = np.linalg.norm(self.ini_hess @ np.array(list(self.ini_est.values())) / self.w_ada, ord=np.inf)

        # optimal lambda value, computed after cross validation
        self.lambda_opt = None

        if penalty is None:
            self.penalty = np.zeros(100)
            self.penalty[1:] = np.exp(np.linspace(start=np.log(0.001), stop=np.log(self.lambda_max), num=99))
            self.penalty[99] = self.lambda_max
        else:
            self.penalty = penalty

        # initialize solution path
        self.est_path = np.empty((len(self.penalty), len(self.ini_est)))
        self.est_path[0] = np.array(list(self.ini_est.values()))

        # details on optim_info results
        self.path_info = []

    def loss(self, param, penalty=1):
        """

        :param param: param at which loss is computed
        :param penalty: constant penalty multiplying the adaptive weights
        :return:
        """
        par = np.array(list(param.values()))
        par_ini = np.array(list(self.ini_est.values()))
        w_a = self.w_ada

        out = 0.5 * (par - par_ini) @ self.ini_hess @ (par - par_ini) + penalty * np.linalg.norm(par * w_a, ord=1)

        return out

    def fit(self, cv=None, **kwargs):

        self.optim_info['args'] = {'cv': cv, **kwargs}

        # in this case start is initial estimate, assumed to be already in model order
        # and the bounds are assumed to be in the same order, so no check on the order is needed


        """

        :param cv: controls validation of the lambda parameter in the lasso path. If none no validation takes place. Otherwise cv must be
        a number in (0,1) controlling the proportion of obs to be used as validation. E.g. if cv = 0.1 the last 10% of
        obs is used as validation and the first 90% as training. Performance on validation set is measured using the loss
        function of the base estimator. Finally the estimate path on the full dataset is computed: the field est
        is filled with the estimate corresponding  to the optimal lambda, which is in turn saved in the field lambda_opt
        :param kwargs:
        :return:
        """
        if cv is None:
            for i in range(len(self.est_path) - 1):
                cur_est = self.proximal_gradient(self.est_path[i], self.penalty[i + 1], **kwargs)
                self.path_info.append(cur_est)
                self.est_path[i + 1] = cur_est['x']
        else:
            n = self.sde.data.n_obs
            n_val = int(cv * n)

            # create auxiliary sde objects
            sde_val = Sde(model=self.sde.model, sampling=self.sde.sampling.sub_sampling(last_n=n_val))
            sde_tr = Sde(model=self.sde.model, sampling=self.sde.sampling.sub_sampling(first_n=n - n_val + 1))

            sde_val.set_data(self.sde.data.data.iloc[-n_val:])
            sde_tr.set_data(self.sde.data.data.iloc[:n-n_val+1])

            # create auxiliary estimator of same type of base est, on training data
            aux_est = type(self.base_est)(sde_tr)
            aux_est.fit(start=self.ini_est, **self.base_est.optim_info['args'])
            lasso_tr = AdaLasso(sde_tr, base_estimator=aux_est, weights=self.weights, delta=self.delta, penalty=self.penalty)
            lasso_tr.fit()

            # create aux est object on validation data and compute loss
            aux_est = type(self.base_est)(sde_val)
            val_loss = np.full(len(lasso_tr.est_path) - 1, np.infty)
            for i in range(len(lasso_tr.est_path) - 1):
                try:
                    val_loss[i] = aux_est.loss(dict(zip(aux_est.sde.model.param, lasso_tr.est_path[i])))
                except:
                    pass

            # compute full path
            self.fit()

            # compute final estimate using optimal lambda
            self.lambda_opt = self.penalty[np.argmin(val_loss)]
            self.est = dict(zip(aux_est.sde.model.param, self.est_path[np.argmin(val_loss)]))
            self.vcov = np.linalg.inv(self.ini_hess)




    def soft_threshold(self, par, penalty):
        return np.sign(par) * np.maximum(np.abs(par) - penalty * self.w_ada, 0)

    def prox_backtrack(self, y_curr, gamma, penalty):

        par_ini = np.array(list(self.ini_est.values()))
        w_a = self.w_ada
        s = 1

        jac_y = self.ini_hess @ (y_curr - par_ini)
        x_curr = self.soft_threshold(y_curr - s * jac_y, penalty * s)

        g_1 = 0.5 * (x_curr - par_ini) @ self.ini_hess @ (x_curr - par_ini)
        g_2 = 0.5 * (y_curr - par_ini) @ self.ini_hess @ (y_curr - par_ini)
        qs_12 = g_2 + (x_curr - y_curr) @ self.ini_hess @ (y_curr - par_ini) \
                + (1 / 2 * s) * np.linalg.norm(x_curr - y_curr, ord=2) ** 2

        if g_1 <= qs_12:
            return s

        while g_1 > qs_12:
            s = gamma * s
            x_curr = self.soft_threshold(y_curr - s * jac_y, penalty)
            qs_12 = g_2 + (x_curr - y_curr) @ self.ini_hess @ (y_curr - par_ini) \
                    + (1 / 2 * s) * np.linalg.norm(x_curr - y_curr, ord=2) ** 2

        return s

    def proximal_gradient(self, x0, penalty, epsilon=1e-03, max_it=1e4, bounds=None, cyclic=False, **kwargs):

        par_ini = np.array(list(self.ini_est.values()))
        w_a = self.w_ada

        t_prev = 1
        x_prev = np.array(x0)
        y_curr = x_prev

        if bounds is not None:
            bounds = np.array(bounds).transpose()
            assert np.all(y_curr > bounds[0]) and np.any(y_curr < bounds[1]), 'starting point outside of bounds'

        jac_y = self.ini_hess @ (y_curr - par_ini)
        padding = np.ones_like(jac_y)

        if cyclic:
            padding = np.zeros_like(jac_y)
            cycle_start = np.argmax(jac_y)
            padding[cycle_start] = 1

        if cyclic:
            s = 1/np.diag(self.ini_hess)[padding == 1]
        else:
            #s = self.prox_backtrack(y_curr, gamma=0.8, penalty=penalty)
            s = 1 / self.lip

        x_curr = self.soft_threshold(y_curr - s * jac_y * padding, penalty * s)

        it_count = 1
        status = 1
        message = ''

        while np.linalg.norm(x_curr - x_prev, ord=2) >= epsilon * s and it_count < max_it:

            if cyclic:
                s = 1/np.diag(self.ini_hess)[padding == 1]
            else:
                #s = self.prox_backtrack(y_curr=y_curr, gamma=0.8, penalty=penalty)
                s = 1/self.lip

            # print('s ' + str(s) + '\n')

            if bounds is not None:
                x_curr[x_curr < bounds[0]] = bounds[0][x_curr < bounds[0]] + epsilon
                x_curr[x_curr > bounds[1]] = bounds[1][x_curr > bounds[1]] - epsilon

            # # interrupt execution before costly evaluation of the gradient
            # if np.linalg.norm(x_curr - x_prev) < 0.1 * epsilon * np.linalg.norm(x_prev):
            #     message = 'Relative reduction less than {0}'.format(0.1*epsilon)
            #     return {'x': x_curr, 'f': f(x_curr, **kwargs), 'status': status, 'message': message, 'niter': it_count, 'jac': jac_y, 'epsilon': epsilon}

            t_curr = (1 + np.sqrt(1 + 4 * t_prev ** 2)) / 2

            y_curr = x_curr + (t_prev - 1) / t_curr * (x_curr - x_prev)

            #
            jac_y = self.ini_hess @ (y_curr - par_ini)
            x_prev = x_curr
            x_curr = self.soft_threshold(y_curr - s * jac_y * padding, penalty * s)

            if bounds is not None:
                y_curr[y_curr < bounds[0]] = bounds[0][y_curr < bounds[0]] + epsilon
                y_curr[y_curr > bounds[1]] = bounds[1][y_curr > bounds[1]] - epsilon

            t_prev = t_curr



            if cyclic:
                # print(padding)
                padding = np.roll(padding, 1)
                if np.argmax(padding) == cycle_start:
                    cycle_start = np.argmax(jac_y)
                    padding = np.zeros_like(jac_y)
                    padding[cycle_start] = 1
            # print(x_curr)
            # print(str(jac_y) + '\n\n')

            it_count += 1

        if np.linalg.norm(x_curr - x_prev, ord=2) >= epsilon * s:
            message = 'Maximum number of iterations reached'
            status = 1
        else:
            message = 'Success: gradient norm less than epsilon'
            status = 0

        return {'x': x_curr, 'f': self.loss_wrap(x_curr, self), 'status': status, 'message': message, 'niter': it_count,
                'jac': jac_y, 'epsilon': epsilon}


    def plot(self):
        plt.figure()
        plt.title('Coefficients path')
        plt.ylabel('Estimates')
        plt.xlabel('log lambda')
        plt.plot(np.log(self.penalty), self.est_path)
        plt.show()
        return