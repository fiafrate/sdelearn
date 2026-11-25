import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import sympy as sym

from .sdelearn import Sde
from .sde_model import SdeModel
from .sde_data import SdeData
from .sde_learner import SdeLearner

import warnings
import scipy as sp
from collections import defaultdict


class Qmle(SdeLearner):
    def __init__(self, sde):
        """
        Create QMLE estimator, a SdeLearner based on quasi-maximum likelihood estimator
        :param sde: object of class Sde
        """
        super().__init__(sde)

        # # contains qml estimate
        # self.est = None
        # # contains var cov matrix
        # self.vcov = None
        # # contains info about optimization
        # self.optim_info = None
        # auxiliary objects: partial computation with current parameter and all x
        self.X = X = sde.data.data.to_numpy().astype('float32', copy=False)
        Xr = X.reshape(sde.data.n_obs, 1, sde.model.n_var)
        self.DX = Xr[1:len(Xr)] - Xr[:len(Xr) - 1]
        self.bs = None
        self.As = None
        self.Ss = None
        self.Ss_inv = None
        self.DXS_inv = None
        # auxiliary value of param: latest value at which derivatives were updated (avoids repetitions)
        self.aux_par = None
        self._aux_group = None
        self.faulty_par = False
        self.batch_id = np.arange(self.sde.data.n_obs - 1)
        self.low_mem = False

    # method either AGD for nesterov otherwise passed on to scipy.optimize, only valid for symbolic mode
    def fit(self, start, method="BFGS", two_step=True, hess_exact=False, **kwargs):
        """

        :param start: optimization starting point
        :param method: optimization method to use: either AGD for accelerated gradient descent (experimental)
            or any method supported by scipy.optimize
        :param two_step: boolean, whether to perform two-step optimization (diffusion first, then drift) as in
        Yoshida, Nakahiro. "Quasi-likelihood analysis and its applications." Statistical Inference for Stochastic Processes 25.1 (2022): 43-60.
        :param hess_exact: use exact computation of the hessian (only in symbolic mode) or use gradient approximation. Available
            only if second derivatives are available in sde.model, otherwise ignored
        :param kwargs: additional parameters passed over to optimizers
        :return: self, after updating self.optim_info, self.est, self.vcov
        """


        # if one of the parameter groups is missing necessarily use two step
        if len(self.sde.model.drift_par) == 0 or len(self.sde.model.diff_par) == 0:
            two_step = True

        # if second derivatives are not available use hess approx
        if self.sde.model.options['hess'] == False:
            hess_exact = False

        # catch args
        self.optim_info['args'] = {'method': method, 'two_step': two_step, 'hess_exact': hess_exact, **kwargs}
        # fix bounds and start order - bounds are assumed to have same order as start!
        if kwargs.get('bounds') is not None:
            bounds = kwargs.get('bounds')
            ik = [list(start.keys()).index(k) for k in self.sde.model.param]
            bounds = [bounds[i] for i in ik]
            kwargs['bounds'] = bounds
        start = {k: start.get(k) for k in self.sde.model.param}

        if self.sde.model.mode == 'fun':
            res = optimize.minimize(fun=self.loss_wrap, x0=np.array(list(start.values())), args=(self), method=method,
                                    **kwargs)
            self.est = dict(zip(self.sde.model.param, res.x))

            if isinstance(res.hess_inv, np.ndarray):
                self.vcov = res.hess_inv / (self.sde.data.n_obs - 1)
                self.optim_info['hess'] = np.linalg.inv(res.hess_inv)
            else:
                self.vcov = res.hess_inv.todense() / (self.sde.data.n_obs - 1)
                self.optim_info['hess'] = np.linalg.inv(self.vcov)

            self.optim_info['res'] = res


        if self.sde.model.mode == 'sym':
            n = self.sde.data.n_obs - 1
            if method == 'AGD':
                x0 = np.array(list(start.values()))
                res = self.nesterov_descent(self.loss_wrap, x0, self.grad_wrap, sde_learn=self, **kwargs)
                self.optim_info['res'] = res
                self.est = dict(zip(self.sde.model.param, res['x']))

            if method != 'AGD':
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                if two_step:
                    # begin two step estimation -------
                    # we are here either because the user chose two_step=True or because
                    # some a parameter group is missing: check whether par group is present
                    # before perfoming partial estimation.

                    # initialize empty result array
                    res_beta = type('res', (), {'x' : np.array([]), 'info': {}})
                    beta_est = np.array([])
                    if len(self.sde.model.diff_par) > 0:
                        # optimize wrt beta (diffusion) -----------------------------------------------
                        # fix bounds
                        kwargs2 = kwargs.copy()
                        if kwargs2.get("bounds") is not None:
                            bounds_id = [list(start.keys()).index(p) for p in self.sde.model.diff_par]
                            kwargs2["bounds"] = [kwargs2.get("bounds")[i] for i in bounds_id]

                        x_start = np.array([start.get(k) for k in self.sde.model.diff_par])
                        group_ini = np.array([start.get(k) for k in self.sde.model.drift_par])

                        diag_diff = self.check_diag_est() and self.sde.model.n_var > 1
                        if diag_diff:
                            # if diagonal matrix perform univariate estimation
                            # NB THIS CURRENTLY WORKS ONLY WHEN DIAGONAL TERMS DEPEND ON 1 VARIABLE ONLY AND THERE ARE NO SHARED PARAMETERS
                            beta_by_name = {}
                            for i in range(self.sde.model.n_var):
                                diff_i = [[sym.simplify(self.sde.model.A_expr[i,i])]]
                                sym_i = set([s.name for s in diff_i[0][0].free_symbols])
                                # for future use, when exogenous variables are added
                                #id_var_i = [j for j in range(self.sde.model.n_var) if self.sde.model.state_var[j] in sym_i and j is not i]
                                state_var_i =[self.sde.model.state_var[i]]
                                mod_i = SdeModel(drift=[0], diff=diff_i, state_var=state_var_i)

                                # perform estimation only if there are some parameters to estimate
                                if len(mod_i.param) > 0:
                                    data_i = SdeData(
                                        self.sde.data.data.iloc[:, i].to_numpy().reshape(self.sde.data.n_obs, 1))
                                    sde_i = Sde(sampling=self.sde.sampling,
                                                model=mod_i,
                                                data=data_i)
                                    qmle_i = Qmle(sde_i)
                                    qmle_i.fit(start, method=method, hess_exact=hess_exact, **kwargs)
                                    beta_by_name.update(qmle_i.est)
                                    res_beta.info[list(qmle_i.est.keys())[0]] = qmle_i.optim_info
                            beta_est = np.array([beta_by_name[name] for name in self.sde.model.diff_par])
                        else:
                            res_beta = \
                                optimize.minimize(fun=self.loss_wrap2,
                                                  x0=x_start,
                                                  args=(self, "beta", group_ini),
                                                  method=method,
                                                  jac=Qmle.grad_wrap2,
                                                  **kwargs2)
                            beta_est = res_beta.x

                    # initialize empty result array
                    res_alpha = type('res', (), {'x' : np.array([]), 'info': {}})
                    alpha_est = np.array([])
                    if len(self.sde.model.drift_par) > 0:
                        # optimize wrt alpha (drift) --------------------------------------------------------
                        # fix bounds
                        self.aux_par = None
                        kwargs2 = kwargs.copy()
                        if kwargs2.get("bounds") is not None:
                            bounds_id = [list(start.keys()).index(p) for p in self.sde.model.drift_par]
                            kwargs2["bounds"] = [kwargs2.get("bounds")[i] for i in bounds_id]

                        x_start = np.array([start.get(k) for k in self.sde.model.drift_par])
                        group_ini = beta_est
                        res_alpha = \
                            optimize.minimize(fun=self.loss_wrap2,
                                              x0=x_start,
                                              args=(self, "alpha", group_ini),
                                              method=method,
                                              jac=Qmle.grad_wrap2,
                                              **kwargs2)
                        alpha_est = res_alpha.x

                    self.est = dict(zip(self.sde.model.param, np.concatenate((alpha_est, beta_est))))
                    # final round of auxiliary quantities updates, in case drift group was missing
                    if len(self.sde.model.drift_par) == 0:
                        self.aux_par = None
                        try:
                            self.update_aux(self.est)
                        except np.linalg.LinAlgError:
                            pass

                    # create result components
                    self.optim_info['res_alpha'] = res_alpha
                    self.optim_info['res_beta'] = res_beta

                    if self.faulty_par:
                        warnings.warn('Singular matrix occurred during optimization. Try a different starting point.\n')

                    # compute and save hessian and vcov, either exact or approx
                    if hess_exact:
                        self.batch_id = np.arange(self.sde.data.n_obs - 1)
                        blocks = [self.hessian(self.est, group=group)
                                  for group, names in (("alpha", self.sde.model.drift_par),
                                                       ("beta", self.sde.model.diff_par)) if names]
                        self.optim_info["hess"] = sp.linalg.block_diag(*blocks)
                    # elif (hasattr(res_alpha, 'hess_inv') or hasattr(res_beta, 'hess_inv')):
                    #     #extract block cov matrix from bfgs or similar if both par blocks are available
                    #     if len(self.sde.model.drift_par) * len(self.sde.model.diff_par) > 0:
                    #         if isinstance(res_alpha.hess_inv, np.ndarray):
                    #             self.vcov = np.block([[res_alpha.hess_inv, np.zeros((len(res_alpha.x), len(res_beta.x)))],
                    #                                   [np.zeros((len(res_beta.x), len(res_alpha.x))), res_beta.hess_inv]]) / (self.sde.data.n_obs - 1)
                    #
                    #         else:
                    #             self.vcov = np.block(
                    #                 [[res_alpha.hess_inv.todense(), np.zeros((len(res_alpha.x), len(res_beta.x)))],
                    #                  [np.zeros((len(res_beta.x), len(res_alpha.x))), res_beta.hess_inv.todense()]]) / (self.sde.data.n_obs - 1)
                    #
                    #         self.optim_info['hess'] = np.linalg.inv(self.vcov * (self.sde.data.n_obs - 1))
                    #
                    #     # extract cov matrix if one param block is missing:
                    #     elif len(self.sde.model.drift_par) > 0:
                    #         if isinstance(res_alpha.hess_inv, np.ndarray):
                    #             self.vcov = res_alpha.hess_inv / (self.sde.data.n_obs - 1)
                    #         else:
                    #             self.vcov = res_alpha.hess_inv.todense() / (self.sde.data.n_obs - 1)
                    #
                    #     elif len(self.sde.model.diff_par) > 0:
                    #         if isinstance(res_beta.hess_inv, np.ndarray):
                    #             self.vcov = res_beta.hess_inv / (self.sde.data.n_obs - 1)
                    #         else:
                    #             self.vcov = res_beta.hess_inv.todense() / (self.sde.data.n_obs - 1)
                    #
                    #         self.optim_info['hess'] = np.linalg.inv(self.vcov * (self.sde.data.n_obs - 1))
                    else:
                        # drift hessian computation
                        if self.sde.model.npar_dr > 0:
                            gs_alpha = self.gradient2(self.est, group='alpha', ret_sample=True)
                            h_alpha = n * (gs_alpha.T @ gs_alpha)
                        # diffusion case: separate case for diagonal estimation
                        if self.sde.model.npar_di > 0:
                            if diag_diff:
                                local_hess = sp.linalg.block_diag(*[v['hess'] for v in res_beta.info.values()])
                                local_positions = {name: i for i, name in enumerate(beta_by_name)}
                                order = [local_positions[name] for name in self.sde.model.diff_par]
                                h_beta = local_hess[np.ix_(order, order)]
                            else:
                                gs_beta = self.gradient2(self.est, group='beta', ret_sample=True)
                                h_beta = n * (gs_beta.T @ gs_beta)

                        #
                        # store results according to groups present
                        if self.sde.model.npar_dr > 0 and self.sde.model.npar_di > 0:
                            self.optim_info["hess"] = np.block([[h_alpha, np.zeros((len(alpha_est), len(beta_est)))],
                                  [np.zeros((len(beta_est), len(alpha_est))), h_beta]])
                        elif self.sde.model.npar_dr > 0:
                            self.optim_info["hess"] = h_alpha
                        else:
                            self.optim_info["hess"] = h_beta




                        #end two-step estimation

                else:
                    # begin simultaneous estimation
                    res = optimize.minimize(fun=self.loss_wrap, x0=np.array(list(start.values())), args=(self),
                                            method=method,
                                            jac=Qmle.grad_wrap, hess=Qmle.hess_wrap, **kwargs)
                    self.est = dict(zip(self.sde.model.param, res.x))

                    self.optim_info['res'] = res
                    if self.faulty_par:
                        warnings.warn('Singular matrix occurred during optimization. Try a different starting point.')

            if method == 'AGD' or not two_step:
                if hess_exact:
                    self.optim_info["hess"] = self.hessian(self.est)
                else:
                    gs = self.gradient(self.est, ret_sample=True)
                    self.optim_info["hess"] = n * (gs.T @ gs)

        # try inverting hessian
        try:
            self.vcov = np.linalg.inv(self.optim_info["hess"])
            if self.sde.model.mode == "sym":
                self.vcov /= n
            else:
                rates = self.rates()
                self.vcov = rates @ self.vcov @ rates
        except np.linalg.LinAlgError:
            warnings.warn(
                'Singular Hessian matrix occurred during optimization. Try a different starting point.\n')


        return self

    def rates(self, group='all'):
        '''
        computes rate matrix, of the type diag(1/sqrt(n delta_n) , 1/sqrt(n))
        :param group: either all, alpha (drift) or beta (diff)
        :return: rates matrix
        '''
        n = self.sde.data.n_obs - 1
        dn = self.sde.sampling.delta
        dr_block_rate = np.eye(self.sde.model.npar_dr) / np.sqrt(n * dn)
        di_block_rate = np.eye(self.sde.model.npar_di) / np.sqrt(n)
        if group == 'all':
            if self.sde.model.npar_dr > 0 and self.sde.model.npar_di > 0:
                rates = sp.linalg.block_diag(dr_block_rate, di_block_rate)
            elif self.sde.model.npar_dr > 0:
                rates = dr_block_rate
            else:
                rates = di_block_rate
        elif group == 'alpha':
            rates = dr_block_rate
        elif group == 'beta':
            rates = di_block_rate
        return rates

    # @staticmethod
    # def qmle(sde, start, upper=None, lower=None):
    #     res = optimize.minimize(fun=Qmle.mlogl, x0=np.array(list(start.values())), args=sde)
    #     out = {"est": res.x,
    #        "vcov": res.hess_inv,
    #        "optim_info": res}
    #     return out

    # @staticmethod
    # def nlogl_f(par, sde):
    #     param = dict(zip(sde.model.param, par))
    #     b = sde.model.drift
    #     A = sde.model.diff
    #
    #
    #     b_n = np.empty((sde.X.shape[0] - 1, sde.model.n_var))
    #     S_n = np.empty((sde.X.shape[0] - 1, sde.model.n_var, sde.model.n_var))
    #
    #     for i in range(sde.X.shape[0] - 1):
    #         b_n[i] = b(sde.X[i], param)
    #         A_temp = A(sde.X[i], param)
    #         S_n[i] = np.matmul(np.array(A_temp), np.array(A_temp).transpose())
    #
    #     X_res = np.array(sde.X[1:] - sde.X[:(sde.X.shape[0] - 1)] - b_n * sde.sampling.delta).reshape(sde.X.shape[0] - 1, sde.model.n_var, 1)
    #     X_res_t = X_res.reshape(sde.X.shape[0] - 1, 1, sde.model.n_var)
    #     sde.debg = {"par": par, "b_n": b_n, "S_n": S_n}
    #     out = 0.5 * np.sum(
    #         np.matmul(X_res_t, np.matmul(np.linalg.inv(S_n), X_res)) * 1 / sde.sampling.delta + np.linalg.slogdet(S_n)[
    #             1])
    #
    #     return out

    def loss(self, param, batch_id=None, **kwargs):
        """Return the negative Gaussian quasi-log-likelihood per increment, up to constants."""
        out = 0
        if self.sde.model.mode == 'fun':
            b = self.sde.model.drift
            A = self.sde.model.diff

            b_n = np.empty((self.X.shape[0] - 1, self.sde.model.n_var))
            S_n = np.empty((self.X.shape[0] - 1, self.sde.model.n_var, self.sde.model.n_var))

            for i in range(self.X.shape[0] - 1):
                b_n[i] = b(self.X[i], param)
                A_temp = A(self.X[i], param)
                S_n[i] = np.matmul(np.array(A_temp), np.array(A_temp).transpose())

            signs, log_dets = np.linalg.slogdet(S_n)
            if np.all(signs > 0):
                delta = self.sde.sampling.delta
                residual = self.X[1:] - self.X[:-1] - b_n * delta
                weighted_residual = np.linalg.solve(S_n, residual[..., None])[..., 0]
                quadratic = np.sum(residual * weighted_residual, axis=1)
                out = 0.5 * np.mean(quadratic / delta + log_dets)
            else:
                out = 1e100

        if self.sde.model.mode == 'sym':

            # check whether both parameter groups are present, otherwise
            # # compute reduced loss
            # if len(self.sde.model.drift_par) == 0:
            #     return self.loss2(param=param, group='beta', batch_id=batch_id, **kwargs)
            #
            # if len(self.sde.model.diff_par) == 0:
            #     return self.loss2(param=param, group='alpha', batch_id=batch_id, **kwargs)


            try:
                self.update_aux(param, batch_id)
            except np.linalg.LinAlgError:
                return (np.random.rand() + 1) * self.X.shape[0] ** 2

            log_dets = np.linalg.slogdet(self.Ss)
            # log_det = np.where(log_dets[0] != 0, log_dets[1], np.zeros_like(log_dets[1]))
            if np.all(log_dets[0] > 0):
                out = 0.5 * np.sum(
                    np.matmul(self.DXS_inv,
                              (self.DX[self.batch_id] - self.sde.sampling.delta * self.bs).transpose(0, 2, 1)).squeeze()
                    * 1/self.sde.sampling.delta
                    + log_dets[1]) * 1 / len(self.batch_id)
            else:
                out = 1e100

        return out

    def loss2(self, param, group, batch_id=None, **kwargs):
        out = 0

        try:
            self.update_aux2(param, group, batch_id)
        except np.linalg.LinAlgError:
            return (np.random.rand() + 1) * self.X.shape[0] ** 2

        if group == "alpha":
            out = 0.5 * np.sum(
                np.matmul(self.DXS_inv,
                          (self.DX[self.batch_id] - self.sde.sampling.delta * self.bs).transpose(0, 2, 1)).squeeze()
                * 1 / self.sde.sampling.delta) * 1 / len(self.batch_id)

        if group == "beta":
            log_dets = np.linalg.slogdet(self.Ss)
            log_det = np.where(log_dets[0] != 0, log_dets[1], np.zeros_like(log_dets[1]))


            if np.all(log_dets[0] > 0):
                out = 0.5 * np.sum(
                    np.matmul(np.matmul(self.DX[self.batch_id], self.Ss_inv),
                              (self.DX[self.batch_id]).transpose(0, 2, 1)).squeeze() * 1 / self.sde.sampling.delta
                    + log_dets[1]) * 1 / len(self.batch_id)
            else:
                out = 1e100

        return out

    # update the auxiliary matrices
    def update_aux(self, param, batch_id=None):
        # skip if derivatives are already updated with this parameter
        indices = np.arange(len(self.DX)) if batch_id is None else np.asarray(batch_id)
        if param == self.aux_par and self._aux_group == "all" and np.array_equal(indices, self.batch_id):
            if self.faulty_par:
                raise np.linalg.LinAlgError
            else:
                return

        self.faulty_par = False
        self._aux_group = "all"
        self.aux_par = param
        self.batch_id = indices

        self.As = self.sde.model.der_foo["A"](*self.X[self.batch_id].transpose(), **param).transpose((2, 0, 1)).astype('float32')
        self.Ss = np.einsum('ndu, neu-> nde', self.As, self.As)

        idx_inv = np.linalg.det(self.Ss) != 0
        if not np.any(idx_inv):
            self.faulty_par = True
            raise np.linalg.LinAlgError

        self.Ss_inv = np.zeros_like(self.Ss)
        self.Ss_inv[idx_inv] = np.linalg.inv(self.Ss[idx_inv])

        self.bs = np.swapaxes(self.sde.model.der_foo["b"](*self.X[self.batch_id].transpose(), **param), 0, -1).astype('float32')

        # self.Ss = np.array([self.model.der_foo["S"](*x, **param) for x in self.X[:-1]])
        # self.bs = np.array([self.model.der_foo["b"](*x, **param) for x in self.X[:-1]]).reshape(
        #     (self.X.shape[0] - 1, 1, self.X.shape[1]))
        self.DXS_inv = np.matmul(self.DX[self.batch_id] - self.sde.sampling.delta * self.bs, self.Ss_inv)

        return

    def update_aux2(self, param, group, batch_id=None, force_recompute=False):
        """
        interally used for updating auxiliary quantities during optimization, like current drift and diffusion values.
        Has checks to avoid recomputing quantities already computed. Controls updated in two step optimization
        :param param: current parameter value
        :param group: group to be optimized, either 'alpha' for drift or 'beta' for diffusion
        :param batch_id: current obs batch, if None all obs are taken into account
        :param force_recompute: recompute updates even if parameter is already encountered. This can happen if only dirft parameters have to be estimated
        :return:
        """
        # skip if derivatives are already updated with this parameter

        indices = np.arange(len(self.DX)) if batch_id is None else np.asarray(batch_id)
        same_batch = np.array_equal(indices, self.batch_id)
        if (not force_recompute and param == self.aux_par and same_batch
                and self._aux_group in ("all", group)):
            if self.faulty_par:
                raise np.linalg.LinAlgError
            return

        if group == "alpha" and (not same_batch or self.aux_par is None or self.faulty_par
                                 or any(param[k] != self.aux_par[k] for k in self.sde.model.diff_par)):
            self.update_aux2(param, "beta", indices, True)

        self._aux_group = group
        self.aux_par = param
        self.faulty_par = False

        self.batch_id = indices

        # these quantities are required in all cases, but if they have been already computed do not compute again

        if group == "beta":
            self.As = self.sde.model.der_foo["A"](*self.X[self.batch_id].transpose(), **param).transpose((2, 0, 1))
            self.Ss = np.einsum('ndu, neu -> nde', self.As, self.As)
            #self.Ss = np.swapaxes(self.sde.model.der_foo["S"](*self.X[self.batch_id].transpose(), **param), 0, -1)
            self.Ss_inv = np.zeros_like(self.Ss)
            idx_inv = np.linalg.det(self.Ss) != 0
            if not np.any(idx_inv):
                self.faulty_par = True
                raise np.linalg.LinAlgError

            self.Ss_inv[idx_inv] = np.linalg.inv(self.Ss[idx_inv])

        if group == "alpha":
            # required only for drift estimation
            self.bs = np.swapaxes(self.sde.model.der_foo["b"](*self.X[self.batch_id].transpose(), **param), 0, -1)
            self.DXS_inv = np.matmul(self.DX[self.batch_id] - self.sde.sampling.delta * self.bs, self.Ss_inv)

        return

    # compute the gradient of the quasi-lik at point par for a given sde object

    def gradient(self, param, batch_id=None, ret_sample=False, asy_scale=False):
        '''
        compute the gradient of the negative quasi-lik at point par for a given sde object
        :param param: dict of parameter values at which evaluate the gradient
        :param batch_id: indices of subset of observation to be used, if None, default, uses all
        :param ret_sample: if True returns the vector of gradient evaluation at each data point, without summing. Defaults to False
        :param asy_scale: use asymptotic rates in scaling, i.e. (1/sqrt(n delta_n), 1/sqrt(n)). If False (default) scale by n_obs, same as the loss
        :return: gradient array with the same size as param. If ret_vec, an array with shape (n_obs, n_param) is returned
        '''
        assert self.sde.model.mode == 'sym', 'Gradient computation available only in symbolic mode'

        dn = self.sde.sampling.delta

        try:
            self.update_aux(param, batch_id)
        except np.linalg.LinAlgError:
            return 10 * np.random.randn(len(self.sde.model.param))

                # DSs = np.array([self.model.der_foo["DS"](*x, **param) for x in self.X[:-1]])

        #DSs = np.moveaxis(self.sde.model.der_foo["DS"](*self.X[self.batch_id].transpose(), **param), -1, 0)


        if self.sde.model.npar_dr > 0:
            # Jbs = np.array([self.model.der_foo["Jb"](*x, **param) for x in self.X[:-1]])
            # split computation for large models, (low memory mode)
            grad_alpha = np.empty((len(self.batch_id), self.sde.model.npar_dr))
            if self.low_mem:
                for i in range(0, len(self.batch_id), 1000):
                    rows = slice(i, i + 1000)
                    batch_indices = self.batch_id[rows]
                    Jbs = np.moveaxis(self.sde.model.der_foo["Jb"](*self.X[batch_indices].transpose(), **param), -1, 0)
                    grad_alpha[rows] = -2 * np.matmul(self.DXS_inv[rows], Jbs)[:, 0, :]
            else:
                batch_indices = self.batch_id
                Jbs = np.moveaxis(self.sde.model.der_foo["Jb"](*self.X[batch_indices].transpose(), **param), -1, 0)
                grad_alpha[:] = -2 * np.matmul(self.DXS_inv, Jbs)[:, 0, :]

            if asy_scale:
                grad_alpha /= np.sqrt(len(self.batch_id) * self.sde.sampling.delta)
            else:
                grad_alpha /= len(self.batch_id)

        if self.sde.model.npar_di > 0 :
            # split computation, in low memory mode. Possibly parallel?
            grad_beta = np.empty((len(self.batch_id), self.sde.model.npar_di))
            #DAs = np.empty((self.sde.data.n_obs - 1, self.sde.model.npar_di, self.sde.model.n_var, self.sde.model.n_noise))
            if self.low_mem:
                for i in range(0, len(self.batch_id), 1000):
                    rows = slice(i, i + 1000)
                    batch_indices = self.batch_id[rows]
                    DAs = np.moveaxis(self.sde.model.der_foo["DA"](*self.X[batch_indices].transpose(), **param), -1, 0)

                    CSs = np.einsum('npdu, neu -> npde', DAs, self.As[rows])
                    DSs = CSs + CSs.transpose((0, 1, 3, 2))

                    GB1 = np.einsum('nde, npef -> npdf', self.Ss_inv[rows], DSs)
                    grad_beta1 = np.trace(GB1, axis1=2, axis2=3)
                    GB2a = np.einsum('npde, nef -> npdf', GB1, self.Ss_inv[rows])
                    grad_beta2 = -1 / dn * np.einsum('npd, nd -> np',
                                                 np.einsum('nd, npde -> npe', (self.DX[batch_indices] - dn * self.bs[rows])[:, 0, :],
                                                           GB2a),
                                                 (self.DX[batch_indices] - dn * self.bs[rows])[:, 0, :])
                    grad_beta[rows] = grad_beta1 + grad_beta2
            else:
                batch_indices = self.batch_id
                DAs = np.moveaxis(self.sde.model.der_foo["DA"](*self.X[batch_indices].transpose(), **param), -1, 0)

                CSs = np.einsum('npdu, neu -> npde', DAs, self.As)
                DSs = CSs + CSs.transpose((0, 1, 3, 2))

                GB1 = np.einsum('nde, npef -> npdf', self.Ss_inv, DSs)
                grad_beta1 = np.trace(GB1, axis1=2, axis2=3)
                GB2a = np.einsum('npde, nef -> npdf', GB1, self.Ss_inv)
                grad_beta2 = -1 / dn * np.einsum('npd, nd -> np',
                                                 np.einsum('nd, npde -> npe',
                                                           (self.DX[batch_indices] - dn * self.bs)[:,
                                                           0, :],
                                                           GB2a),
                                                 (self.DX[batch_indices] - dn * self.bs)[:, 0, :])
                grad_beta[:] = grad_beta1 + grad_beta2

            if asy_scale:
                grad_beta /= np.sqrt(len(self.batch_id))
            else:
                grad_beta /= len(self.batch_id)

        if self.sde.model.npar_dr > 0 and  self.sde.model.npar_di > 0:
            if ret_sample:
                out = 0.5 * np.concatenate([grad_alpha, grad_beta], axis=1)
            else:
                out = 0.5 * np.concatenate([np.sum(grad_alpha, axis=0), np.sum(grad_beta, axis=0)])
        elif self.sde.model.npar_dr > 0:
            if ret_sample:
                out = 0.5 * grad_alpha
            else:
                out = 0.5 * np.sum(grad_alpha, axis=0)
        else:
            if ret_sample:
                out = 0.5 * grad_beta
            else:
                out= 0.5 * np.sum(grad_beta, axis=0)


        return out





    def gradient2(self, param, group, batch_id=None, ret_sample=False, asy_scale=False):
        """
        Function for evaluating the gradient for a specific group of parameters, based on 'adaptive' losses, i.e.
        two-step estimation procedures.

        :param param: param dict a which to evaluate the gradient
        :param group: string specifying the parameter group, 'alpha' (drift) or 'beta' (diffusion)
        :param batch_id: internally used for stochastic GD
        :param ret_sample: if True, returns gradient evaluation for each sample. Used in OPG Hessian approximation
        :param asy_scale: use asymptotic rates in scaling, i.e. (1/sqrt(n delta_n), 1/sqrt(n)). If False (default) scale by n_obs, same as the loss
        :return: gradient with respect to drift or diffusion parameter separately
        """

        # assert self.sde.model.mode == 'sym', 'Gradient computation available only in symbolic mode'

        dn = self.sde.sampling.delta

        if group == "alpha":
            try:
                self.update_aux2(param, "alpha", batch_id)
            except np.linalg.LinAlgError:
                return 10 * np.random.randn(len(self.sde.model.drift_par))

            # low memory computation for large models, avoids ram issues
            grad_alpha = np.empty((len(self.batch_id), self.sde.model.npar_dr))
            if self.low_mem:
                for i in range(0, len(self.batch_id), 1000):
                    rows = slice(i, i + 1000)
                    batch_indices = self.batch_id[rows]
                    Jbs = np.moveaxis(self.sde.model.der_foo["Jb"](*self.X[batch_indices].transpose(), **param), -1, 0)
                    grad_alpha[rows] = -2 * np.matmul(self.DXS_inv[rows], Jbs)[:, 0, :]
            else:
                batch_indices = self.batch_id
                Jbs = np.moveaxis(self.sde.model.der_foo["Jb"](*self.X[batch_indices].transpose(), **param), -1, 0)
                grad_alpha[:] = -2 * np.matmul(self.DXS_inv, Jbs)[:, 0, :]

            if asy_scale:
                grad_alpha /= np.sqrt(len(self.batch_id) * self.sde.sampling.delta)
            else:
                grad_alpha /= len(self.batch_id)

            if ret_sample:
                return 0.5 * grad_alpha
            else:
                return 0.5 * np.sum(grad_alpha, axis=0)


        if group == "beta":
            try:
                self.update_aux2(param, "beta", batch_id)
            except np.linalg.LinAlgError:
                return 10 * np.random.randn(len(self.sde.model.diff_par))

            # DSs = np.moveaxis(self.sde.model.der_foo["DS"](*self.X[self.batch_id].transpose(), **param), -1, 0)
            grad_beta = np.empty((len(self.batch_id), self.sde.model.npar_di))
            if self.low_mem:
                for i in range(0, len(self.batch_id), 1000):
                    rows = slice(i, i + 1000)
                    batch_indices = self.batch_id[rows]
                    DAs = np.moveaxis(
                        self.sde.model.der_foo["DA"](*self.X[batch_indices].transpose(), **param), -1, 0)
                    CSs = np.einsum('npdu, neu -> npde', DAs, self.As[rows])
                    DSs = CSs + CSs.transpose((0, 1, 3, 2))
                    GB1 = np.einsum('nde, npef -> npdf', self.Ss_inv[rows], DSs)
                    grad_beta1 = np.trace(GB1, axis1=2, axis2=3)
                    GB2a = np.einsum('npde, nef -> npdf', GB1, self.Ss_inv[rows])
                    grad_beta2 = -1 / dn * np.einsum('npd, nd -> np',
                                                     np.einsum('nd, npde -> npe', (self.DX[batch_indices])[:, 0, :], GB2a),
                                                     (self.DX[batch_indices])[:, 0, :])
                    grad_beta[rows] = grad_beta1 + grad_beta2
            else:
                batch_indices = self.batch_id
                DAs = np.moveaxis(
                    self.sde.model.der_foo["DA"](*self.X[batch_indices].transpose(), **param), -1, 0)
                CSs = np.einsum('npdu, neu -> npde', DAs, self.As)
                DSs = CSs + CSs.transpose((0, 1, 3, 2))
                GB1 = np.einsum('nde, npef -> npdf', self.Ss_inv, DSs)
                grad_beta1 = np.trace(GB1, axis1=2, axis2=3)
                GB2a = np.einsum('npde, nef -> npdf', GB1, self.Ss_inv)
                grad_beta2 = -1 / dn * np.einsum('npd, nd -> np',
                                                 np.einsum('nd, npde -> npe', (self.DX[batch_indices])[:, 0, :], GB2a),
                                                 (self.DX[batch_indices])[:, 0, :])
                grad_beta[:] = grad_beta1 + grad_beta2

            if asy_scale:
                grad_beta /= np.sqrt(len(self.batch_id))
            else:
                grad_beta /= len(self.batch_id)

            if ret_sample:
                return 0.5 * grad_beta
            else:
                return 0.5 * np.sum(grad_beta, axis=0)

        return None

    # def gbeta(self, param):
    #     dn = self.sde.sampling.delta
    #     out = np.zeros(self.sde.model.npar_di)
    #     for i in range(0, len(self.batch_id)):
    #         DAs = self.sde.model.der_foo["DA"](*self.X[i], **param)
    #         CSs = np.einsum('pdu, eu -> pde', DAs, self.As[i])
    #         DSs = CSs + CSs.transpose((0, 2, 1))
    #         GB1 = np.einsum('de, pef -> pdf', self.Ss_inv[i], DSs)
    #         grad_beta1 = np.trace(GB1, axis1=1, axis2=2)
    #         GB2a = np.einsum('pde, ef -> pdf', GB1, self.Ss_inv[i])
    #         grad_beta2 = -1 / dn * np.einsum('pd, d -> p',
    #                                          np.einsum('d, pde -> pe', (self.DX[i])[0, :], GB2a),
    #                                          (self.DX[i])[0, :])
    #         out += grad_beta1 + grad_beta2
    #     return out




    # compute the hessian of the quasi-lik at point par for a given sde object
    def hessian(self, param, batch_id=None, group="all", **kwargs):
        """Exact Hessian of the mean symbolic quasi-loss on the selected increments.

        "all" uses drift-adjusted residuals and includes mixed derivatives.
        "alpha" and "beta" differentiate the corresponding two-step objectives.
        The beta step uses raw increments. Rows and columns follow model parameter
        order within the selected group. An empty group returns a (0, 0) matrix.
        Requires model option hess=True for the requested second derivatives.
        """
        assert self.sde.model.mode == "sym", "Hessian computation available only in symbolic mode"
        if group not in ("all", "alpha", "beta"):
            raise ValueError("group must be 'all', 'alpha' or 'beta'")
        model = self.sde.model
        drift = group != "beta" and model.npar_dr > 0
        diffusion = group != "alpha" and model.npar_di > 0
        size = (model.npar_dr if drift else 0) + (model.npar_di if diffusion else 0)
        if size == 0:
            return np.empty((0, 0))

        indices = np.arange(len(self.DX)) if batch_id is None else np.asarray(batch_id)
        try:
            if group == "beta":
                self.update_aux2(param, "beta", indices)
            else:
                self.update_aux(param, indices)
        except np.linalg.LinAlgError:
            return np.eye(size) * np.finfo(float).resolution

        def derivative(name):
            values = model.der_foo[name](*self.X[indices].T, **param)
            return np.moveaxis(values, -1, 0)

        dt = self.sde.sampling.delta
        precision = self.Ss_inv
        if group == "beta":
            weighted_residual = (self.DX[indices] @ precision)[:, 0, :]
        else:
            weighted_residual = self.DXS_inv[:, 0, :]

        if drift:
            Jb = derivative("Jb")
            weighted_Jb = precision @ Jb
            alpha = dt * np.einsum("nia,nib->ab", Jb, weighted_Jb)
            alpha -= np.einsum("ni,niab->ab", weighted_residual, derivative("Hb"))
            alpha /= len(indices)
            if not diffusion:
                return alpha

        DA = derivative("DA")
        product = DA @ self.As[:, None].swapaxes(-1, -2)
        DS = product + product.swapaxes(-1, -2)
        second = derivative("HA") @ self.As[:, None, None].swapaxes(-1, -2)
        second += np.einsum("npik,nqjk->npqij", DA, DA)
        HS = second + second.swapaxes(-1, -2)

        # Second derivatives of log(det(S)) and of the residual quadratic form.
        weighted_DS = precision[:, None] @ DS
        beta = np.einsum("nij,npqji->pq", precision, HS)
        beta -= np.einsum("npij,nqji->pq", weighted_DS, weighted_DS)
        v = np.einsum("npij,nj->npi", DS, weighted_residual)
        quadratic = 2 * np.einsum("npi,nij,nqj->pq", v, precision, v, optimize=True)
        quadratic -= np.einsum("ni,npqij,nj->pq", weighted_residual, HS, weighted_residual)
        beta = (beta + quadratic / dt) / (2 * len(indices))
        if not drift:
            return beta

        mixed = np.einsum("nia,npi->ap", weighted_Jb, v) / len(indices)
        return np.block([[alpha, mixed], [mixed.T, beta]])

    def check_diag_est(self):

        A = self.sde.model.A_expr
        if A.shape[0] != A.shape[1]:
            return False
        is_diagonal = all(sym.simplify(A[i, j]) == 0
                          for i in range(A.shape[0]) for j in range(A.shape[1]) if i != j)
        if is_diagonal:
            from collections import defaultdict
            # d maps every parameter to the equations it appears in
            d = defaultdict(list)
            tuples = [(vi, k) for k, v in self.sde.model.par_map_diff.items() for vi in v]
            for k, v in tuples:
                d[k].append(v)

            one_var_eq = np.all([v == set([k]) or v == set() for k, v in self.sde.model.var_map_diff.items()])
            one_par_eq = np.all([len(v) == 1 for k, v in d.items()])

            return one_var_eq and one_par_eq
        else:
            return False

    def set_low_mem(self, switch=False):
        '''
        uses low memory mode. Reduces memory usage by splitting the gradient computation in batches.
        Recommended for large models, when SIGKILL due to RAM exceedence might occur.
        :param switch: boolean, sets low_memory mode.
        :return: self
        '''
        self.low_mem = switch
        return self

    @staticmethod
    def loss_wrap2(par, sde_learn, group, group_ini, **kwargs):
        """
        wrapping function for optimizers
        :param par:
        :param sde_learn:
        :return:
        """
        if group == "alpha":
            par = np.concatenate((par, group_ini))
        if group == "beta":
            par = np.concatenate((group_ini, par))
        param = dict(zip(sde_learn.sde.model.param, par))
        return sde_learn.loss2(param=param, group=group, **kwargs)

    @staticmethod
    def grad_wrap2(par, sde_learn, group, group_ini, **kwargs):
        """
       wrapping function for optimizers
       :param par:
       :param sde_learn:
       :return:
       """
        if group == "alpha":
            par = np.concatenate((par, group_ini))
        if group == "beta":
            par = np.concatenate((group_ini, par))
        param = dict(zip(sde_learn.sde.model.param, par))
        return sde_learn.gradient2(param=param, group=group, **kwargs)

