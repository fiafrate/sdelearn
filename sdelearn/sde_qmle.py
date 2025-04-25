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
            if method == 'AGD':
                x0 = np.array(list(start.values()))
                res = self.nesterov_descent(self.loss_wrap, x0, self.grad_wrap, sde_learn=self, **kwargs)
                # bounds=kwargs.get('bounds'),
                # cyclic=False if kwargs.get('cyclic') is None else kwargs.get('cyclic'),
                # sde_learn=self)

                self.optim_info['res'] = res
                self.optim_info['hess'] = self.hessian(self.est)

                self.est = dict(zip(self.sde.model.param, res['x']))
                self.vcov = np.linalg.inv(self.optim_info['hess']) / (self.sde.data.n_obs - 1)


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
                            beta_est = np.empty(self.sde.model.npar_di)
                            it_ = 0
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
                                    beta_est[it_] = list(qmle_i.est.values())[0]
                                    it_ += 1
                                    res_beta.info[list(qmle_i.est.keys())[0]] = qmle_i.optim_info
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
                        self.update_aux(self.est)
                    # create result components
                    self.optim_info['res_alpha'] = res_alpha
                    self.optim_info['res_beta'] = res_beta

                    if self.faulty_par:
                        warnings.warn('Singular matrix occurred during optimization. Try a different starting point.\n')

                    # compute and save hessian and vcov, either exact or approx
                    if hess_exact:
                        self.batch_id = np.arange(self.sde.data.n_obs - 1)
                        self.optim_info["hess"] = self.hessian(self.est)
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
                            gs_alpha = self.gradient2(self.est, group='alpha', ret_sample=True, asy_scale=True)
                            h_alpha = np.tensordot(gs_alpha, gs_alpha, (0, 0))
                        # diffusion case: separate case for diagonal estimation
                        if self.sde.model.npar_di > 0:
                            if diag_diff:
                                h_beta = sp.linalg.block_diag(*[v['hess'] for k, v in res_beta.info.items()])
                                # stop here
                            else:
                                gs_beta = self.gradient2(self.est, group='beta', ret_sample=True, asy_scale=True)
                                h_beta = np.tensordot(gs_beta, gs_beta, (0, 0))

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

                    if hess_exact:
                        self.batch_id = np.arange(self.sde.data.n_obs - 1)
                        self.optim_info["hess"] = self.hessian(self.est)
                    # else:
                    #     if isinstance(res.hess_inv, np.ndarray):
                    #         self.vcov = res.hess_inv / (self.sde.data.n_obs - 1)
                    #     else:
                    #         self.vcov = res.hess_inv.todense() / (self.sde.data.n_obs - 1)
                    #     self.optim_info['hess'] = np.linalg.inv(self.vcov * (self.sde.data.n_obs - 1))
                    else:
                        gs = self.gradient(self.est, ret_sample=True, asy_scale=True)
                        self.optim_info["hess"] = np.tensordot(gs, gs, (0,0))


                    self.optim_info['res'] = res
                    if self.faulty_par:
                        warnings.warn('Singular matrix occurred during optimization. Try a different starting point.\n')

                    # end simultaneous estimation

        # try inverting hessian
        try:
            self.vcov = np.linalg.inv(self.optim_info["hess"])
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

            X_res = np.array(self.X[1:] - self.X[:(self.X.shape[0] - 1)] - b_n * self.sde.sampling.delta).reshape(
                self.X.shape[0] - 1, self.sde.model.n_var, 1)
            X_res_t = X_res.reshape(self.X.shape[0] - 1, 1, self.sde.model.n_var)
            log_dets = np.linalg.slogdet(S_n)
            if np.all(log_dets[0] > 0):
                out = 0.5 * np.sum(
                    np.matmul(X_res_t, np.matmul(np.linalg.inv(S_n), X_res)) * 1 / self.sde.sampling.delta +
                    log_dets[1])
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
        if param == self.aux_par:
            if self.faulty_par:
                raise np.linalg.LinAlgError
            else:
                return

        self.aux_par = param
        if batch_id is not None:
            self.batch_id = batch_id
        else:
            self.batch_id = np.arange(self.sde.data.n_obs - 1)

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

        self.faulty_par = False
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

        if param == self.aux_par:
            if self.faulty_par:
                raise np.linalg.LinAlgError
            elif not force_recompute:
                return

        self.aux_par = param
        if batch_id is not None:
            self.batch_id = batch_id
        else:
            self.batch_id = np.arange(self.sde.data.n_obs - 1)

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
            if self.Ss_inv is None:
                self.update_aux2(param, 'beta', batch_id, True)
            self.bs = np.swapaxes(self.sde.model.der_foo["b"](*self.X[self.batch_id].transpose(), **param), 0, -1)
            self.DXS_inv = np.matmul(self.DX[self.batch_id] - self.sde.sampling.delta * self.bs, self.Ss_inv)

        self.faulty_par = False
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
            grad_alpha = np.empty((self.sde.data.n_obs - 1, self.sde.model.npar_dr))
            if self.low_mem:
                for i in range(0, len(self.batch_id), 1000):
                    batch_indices = self.batch_id[i:i + 1000]
                    Jbs = np.moveaxis(self.sde.model.der_foo["Jb"](*self.X[batch_indices].transpose(), **param), -1, 0)
                    grad_alpha[batch_indices] = -2 * np.matmul(self.DXS_inv[batch_indices], Jbs)[:, 0, :]
            else:
                batch_indices = self.batch_id
                Jbs = np.moveaxis(self.sde.model.der_foo["Jb"](*self.X[batch_indices].transpose(), **param), -1, 0)
                grad_alpha[batch_indices] = -2 * np.matmul(self.DXS_inv[batch_indices], Jbs)[:, 0, :]

            if asy_scale:
                grad_alpha /= np.sqrt(len(self.batch_id) * self.sde.sampling.delta)
            else:
                grad_alpha /= len(self.batch_id)

        if self.sde.model.npar_di > 0 :
            # split computation, in low memory mode. Possibly parallel?
            grad_beta = np.empty((self.sde.data.n_obs - 1, self.sde.model.npar_di))
            #DAs = np.empty((self.sde.data.n_obs - 1, self.sde.model.npar_di, self.sde.model.n_var, self.sde.model.n_noise))
            if self.low_mem:
                for i in range(0, len(self.batch_id), 1000):
                    batch_indices = self.batch_id[i:i + 1000]
                    DAs = np.moveaxis(self.sde.model.der_foo["DA"](*self.X[batch_indices].transpose(), **param), -1, 0)

                    CSs = np.einsum('npdu, neu -> npde', DAs, self.As[batch_indices])
                    DSs = CSs + CSs.transpose((0, 1, 3, 2))

                    GB1 = np.einsum('nde, npef -> npdf', self.Ss_inv[batch_indices], DSs)
                    grad_beta1 = np.trace(GB1, axis1=2, axis2=3)
                    GB2a = np.einsum('npde, nef -> npdf', GB1, self.Ss_inv[batch_indices])
                    grad_beta2 = -1 / dn * np.einsum('npd, nd -> np',
                                                 np.einsum('nd, npde -> npe', (self.DX[batch_indices] - dn * self.bs[batch_indices])[:, 0, :],
                                                           GB2a),
                                                 (self.DX[batch_indices] - dn * self.bs[batch_indices])[:, 0, :])
                    grad_beta[batch_indices] = grad_beta1 + grad_beta2
            else:
                batch_indices = self.batch_id
                DAs = np.moveaxis(self.sde.model.der_foo["DA"](*self.X[batch_indices].transpose(), **param), -1, 0)

                CSs = np.einsum('npdu, neu -> npde', DAs, self.As[batch_indices])
                DSs = CSs + CSs.transpose((0, 1, 3, 2))

                GB1 = np.einsum('nde, npef -> npdf', self.Ss_inv[batch_indices], DSs)
                grad_beta1 = np.trace(GB1, axis1=2, axis2=3)
                GB2a = np.einsum('npde, nef -> npdf', GB1, self.Ss_inv[batch_indices])
                grad_beta2 = -1 / dn * np.einsum('npd, nd -> np',
                                                 np.einsum('nd, npde -> npe',
                                                           (self.DX[batch_indices] - dn * self.bs[batch_indices])[:,
                                                           0, :],
                                                           GB2a),
                                                 (self.DX[batch_indices] - dn * self.bs[batch_indices])[:, 0, :])
                grad_beta[batch_indices] = grad_beta1 + grad_beta2

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
            grad_alpha = np.empty((self.sde.data.n_obs-1, self.sde.model.npar_dr))
            if self.low_mem:
                for i in range(0, len(self.batch_id), 1000):
                    batch_indices = self.batch_id[i:i + 1000]
                    Jbs = np.moveaxis(self.sde.model.der_foo["Jb"](*self.X[batch_indices].transpose(), **param), -1, 0)
                    grad_alpha[batch_indices] = -2 * np.matmul(self.DXS_inv[batch_indices], Jbs)[:, 0, :]
            else:
                batch_indices = self.batch_id
                Jbs = np.moveaxis(self.sde.model.der_foo["Jb"](*self.X[batch_indices].transpose(), **param), -1, 0)
                grad_alpha[batch_indices] = -2 * np.matmul(self.DXS_inv[batch_indices], Jbs)[:, 0, :]

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
            grad_beta = np.empty((self.sde.data.n_obs - 1, self.sde.model.npar_di))
            if self.low_mem:
                for i in range(0, len(self.batch_id), 1000):
                    batch_indices = self.batch_id[i:i + 1000]
                    DAs = np.moveaxis(
                        self.sde.model.der_foo["DA"](*self.X[batch_indices].transpose(), **param), -1, 0)
                    CSs = np.einsum('npdu, neu -> npde', DAs, self.As[batch_indices])
                    DSs = CSs + CSs.transpose((0, 1, 3, 2))
                    GB1 = np.einsum('nde, npef -> npdf', self.Ss_inv[batch_indices], DSs)
                    grad_beta1 = np.trace(GB1, axis1=2, axis2=3)
                    GB2a = np.einsum('npde, nef -> npdf', GB1, self.Ss_inv[batch_indices])
                    grad_beta2 = -1 / dn * np.einsum('npd, nd -> np',
                                                     np.einsum('nd, npde -> npe', (self.DX[batch_indices])[:, 0, :], GB2a),
                                                     (self.DX[batch_indices])[:, 0, :])
                    grad_beta[batch_indices] = grad_beta1 + grad_beta2
            else:
                batch_indices = self.batch_id
                DAs = np.moveaxis(
                    self.sde.model.der_foo["DA"](*self.X[batch_indices].transpose(), **param), -1, 0)
                CSs = np.einsum('npdu, neu -> npde', DAs, self.As[batch_indices])
                DSs = CSs + CSs.transpose((0, 1, 3, 2))
                GB1 = np.einsum('nde, npef -> npdf', self.Ss_inv[batch_indices], DSs)
                grad_beta1 = np.trace(GB1, axis1=2, axis2=3)
                GB2a = np.einsum('npde, nef -> npdf', GB1, self.Ss_inv[batch_indices])
                grad_beta2 = -1 / dn * np.einsum('npd, nd -> np',
                                                 np.einsum('nd, npde -> npe', (self.DX[batch_indices])[:, 0, :], GB2a),
                                                 (self.DX[batch_indices])[:, 0, :])
                grad_beta[batch_indices] = grad_beta1 + grad_beta2

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
    def hessian(self, param, batch_id=None, **kwargs):

        assert self.sde.model.mode == 'sym', 'Hessian computation available only in symbolic mode'

        try:
            self.update_aux(param, batch_id)
        except np.linalg.LinAlgError:
            return np.diag(np.full(shape=len(self.sde.model.param), fill_value=np.finfo(float).resolution))

        dn = self.sde.sampling.delta

        Jbs = np.moveaxis(self.sde.model.der_foo["Jb"](*self.X[self.batch_id].transpose(), **param), -1, 0)
        # DSs = np.array([self.model.der_foo["DS"](*x, **param) for x in self.X[:-1]])
        # DSs = np.moveaxis(self.sde.model.der_foo["DS"](*self.X[self.batch_id].transpose(), **param), -1, 0)
        DAs = np.moveaxis(self.sde.model.der_foo["DA"](*self.X[self.batch_id].transpose(), **param), -1, 0)
        CSs = np.einsum('npdu, neu -> npde', DAs, self.As)
        DSs = CSs + CSs.transpose((0, 1, 3, 2))


        # Hbs = np.array([self.model.der_foo["Hb"](*x, **param) for x in self.X[:-1]])
        # HSs = np.array([self.model.der_foo["HS"](*x, **param) for x in self.X[:-1]])
        Hbs = np.moveaxis(self.sde.model.der_foo["Hb"](*self.X[self.batch_id].transpose(), **param), -1, 0)
        #HSs = np.moveaxis(self.sde.model.der_foo["HS"](*self.X[self.batch_id].transpose(), **param), -1, 0)
        HAs = np.moveaxis(self.sde.model.der_foo["HA"](*self.X[self.batch_id].transpose(), **param), -1, 0)
        E1s = np.einsum('npqdu, neu -> npqde', HAs, self.As)
        E2s = np.einsum('npdu, nqeu -> npqde', DAs, DAs)
        HSs = E1s + E2s + E1s.transpose((0,1,2,4,3)) + E2s.transpose((0,1,2,4,3))


        if len(self.sde.model.drift_par) > 0:
            HESSA1 = np.einsum('nd, ndpq -> npq', self.DXS_inv[:, 0, :], Hbs)
            HESSA2 = np.einsum('npd, ndq -> npq', Jbs.transpose((0, 2, 1)), np.einsum('nde, nep -> ndp', self.Ss_inv, Jbs))
            hess_alpha = np.sum(-2 * HESSA1 + 2 * dn * HESSA2, axis=0)


        if len(self.sde.model.diff_par) > 0:
            GB1 = np.einsum('nde, npef -> npdf', self.Ss_inv, DSs)
            GB2a = np.einsum('npde, nef -> npdf', GB1, self.Ss_inv)

            HESSO1 = np.einsum('npde, nqef -> npqdf', GB1, GB1)
            HESS02 = np.einsum('nde, npqef -> npqdf', self.Ss_inv, HSs)

            HESSB1 = np.trace(HESSO1, axis1=3, axis2=4)
            HESSB2 = np.trace(HESS02, axis1=3, axis2=4)

            #HESSB3a = np.einsum('npde, nef, nqfg -> npqdg', DSs, self.Ss_inv, DSs)
            HESSB3a = np.einsum('npqde, nef -> npqdf', HESSO1, self.Ss_inv)
            HESSB3b = np.einsum('npqde, nef -> npqdf', HESS02, self.Ss_inv)
            #HESSB3c = np.einsum('nde, npqef, nfg -> npqdg', self.Ss_inv, HESSB3a - HSs + HESSB3a.transpose((0, 2, 1, 3, 4)), self.Ss_inv)
            HESSB3c = HESSB3a + HESSB3b + HESSB3a.transpose((0, 2, 1, 3, 4))
            HESSB3 = np.einsum('nd, npqde, ne -> npq', self.DXS_inv[:, 0, :], HESSB3c, self.DXS_inv[:, 0, :])

            hess_beta = - HESSB1 + HESSB2 + (1/dn) * HESSB3
            # make sure result is symmetric
            #hess_beta = np.sum(0.5 * (hess_beta + hess_beta.transpose((0, 2, 1))), axis=0)
            hess_beta = np.sum(hess_beta , axis=0)

        if len(self.sde.model.diff_par) * len(self.sde.model.drift_par)> 0:
            hess_ab = -2 * np.sum(
                np.einsum('nd, npde, neq -> npq', (self.DX[self.batch_id] - dn * self.bs)[:, 0, :], GB2a, Jbs), axis=0)

        if len(self.sde.model.diff_par) * len(self.sde.model.drift_par)> 0:
           hess = 0.5 * np.block([[hess_alpha, hess_ab.transpose()], [hess_ab, hess_beta]]) * 1 / len(self.batch_id)

        if len(self.sde.model.diff_par) == 0:
            hess = hess_alpha
        if len(self.sde.model.drift_par) == 0:
            hess = hess_beta

        return hess


    def check_diag_est(self):

        S_sim = np.array(sym.simplify(self.sde.model.S_expr))
        is_diagonal = np.all(S_sim == np.diag(np.diagonal(S_sim)))
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

