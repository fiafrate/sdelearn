import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

from .sdelearn import Sde
from .sde_learner import SdeLearner

import warnings


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
        self.X = X = sde.data.data.to_numpy()
        Xr = X.reshape(sde.data.n_obs, 1, sde.model.n_var)
        self.DX = Xr[1:len(Xr)] - Xr[:len(Xr) - 1]
        self.bs = None
        self.Ss = None
        self.Ss_inv = None
        self.DXS_inv = None
        # auxiliary value of param: latest value at which derivatives were updated (avoids repetitions)
        self.aux_par = None
        self.batch_id = np.arange(self.sde.data.n_obs - 1)


    # method either AGD for nesterov otherwise passed on to scipy.optimize, only valid for symbolic mode
    def fit(self, start, method="BFGS", **kwargs):

        # catch args
        self.optim_info['args'] = {'method': method, **kwargs}
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
            if method == 'BFGS':
                self.vcov = res.hess_inv
            elif method == 'L-BFGS-B':
                self.vcov = res.hess_inv.todense()
            else:
                self.vcov = np.linalg.inv(self.hessian(self.est))

            self.optim_info['res'] = res
            return self

        if self.sde.model.mode == 'sym':
            if method == 'AGD':
                x0 = np.array(list(start.values()))
                res = self.nesterov_descent(self.loss_wrap, x0, self.grad_wrap, sde_learn=self, **kwargs)
                # bounds=kwargs.get('bounds'),
                # cyclic=False if kwargs.get('cyclic') is None else kwargs.get('cyclic'),
                # sde_learn=self)
                self.est = dict(zip(self.sde.model.param, res['x']))
                self.vcov = np.linalg.inv(self.hessian(self.est))
                self.optim_info['res'] = res
                return self

            if method != 'AGD':
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                res = optimize.minimize(fun=self.loss_wrap, x0=np.array(list(start.values())), args=(self),
                                        method=method,
                                        jac=Qmle.grad_wrap, hess=Qmle.hess_wrap, **kwargs)
                self.est = dict(zip(self.sde.model.param, res.x))
                if method == 'BFGS':
                    self.vcov = res.hess_inv
                elif method == 'L-BFGS-B':
                    self.vcov = res.hess_inv.todense()
                else:
                    self.vcov = np.linalg.inv(self.hessian(self.est))
                self.optim_info['res'] = res
                return self

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
            out = 0.5 * np.sum(
                np.matmul(X_res_t, np.matmul(np.linalg.inv(S_n), X_res)) * 1 / self.sde.sampling.delta +
                np.linalg.slogdet(S_n)[
                    1])

        if self.sde.model.mode == 'sym':
            try:
                self.update_aux(param, batch_id)
            except np.linalg.LinAlgError:
                return np.finfo(float).max
            out = 0.5 * np.sum(
                np.matmul(self.DXS_inv,
                          (self.DX[self.batch_id] - self.sde.sampling.delta * self.bs).transpose(0, 2,
                                                                                  1)).squeeze() * 1 / self.sde.sampling.delta
                + np.linalg.slogdet(self.Ss)[1]) * 1 / len(self.batch_id)

        return out

    # update the auxiliary matrices
    def update_aux(self, param, batch_id=None):
        # skip if derivatives are already updated with this parameter
        if param == self.aux_par:
            return

        self.aux_par = param
        if batch_id is not None:
            self.batch_id = batch_id
        else:
            self.batch_id = np.arange(self.sde.data.n_obs - 1)


        self.Ss = np.swapaxes(self.sde.model.der_foo["S"](*self.X[self.batch_id].transpose(), **param), 0, -1)
        self.bs = np.swapaxes(self.sde.model.der_foo["b"](*self.X[self.batch_id].transpose(), **param), 0, -1)

        # self.Ss = np.array([self.model.der_foo["S"](*x, **param) for x in self.X[:-1]])
        # self.bs = np.array([self.model.der_foo["b"](*x, **param) for x in self.X[:-1]]).reshape(
        #     (self.X.shape[0] - 1, 1, self.X.shape[1]))
        self.Ss_inv = np.linalg.inv(self.Ss)
        self.DXS_inv = np.matmul(self.DX[self.batch_id] - self.sde.sampling.delta * self.bs, self.Ss_inv)

        return

    # compute the gradient of the quasi-lik at point par for a given sde object

    def gradient(self, param, batch_id=None):

        assert self.sde.model.mode == 'sym', 'Gradient computation available only in symbolic mode'

        dn = self.sde.sampling.delta

        try:
            self.update_aux(param, batch_id)
        except np.linalg.LinAlgError:
            return np.full(shape=len(self.sde.model.param), fill_value=np.finfo(float).max)

        # Jbs = np.array([self.model.der_foo["Jb"](*x, **param) for x in self.X[:-1]])
        Jbs = np.moveaxis(self.sde.model.der_foo["Jb"](*self.X[self.batch_id].transpose(), **param), -1, 0)
        # DSs = np.array([self.model.der_foo["DS"](*x, **param) for x in self.X[:-1]])
        DSs = np.moveaxis(self.sde.model.der_foo["DS"](*self.X[self.batch_id].transpose(), **param), -1, 0)

        grad_alpha = -2 * np.matmul(self.DXS_inv, Jbs)[:, 0, :]
        GB1 = np.einsum('nde, npef -> npdf', self.Ss_inv, DSs)
        grad_beta1 = np.trace(GB1, axis1=2, axis2=3)
        GB2a = np.einsum('npde, nef -> npdf', GB1, self.Ss_inv)
        grad_beta2 = -1 / dn * np.einsum('npd, nd -> np',
                                         np.einsum('nd, npde -> npe', (self.DX[self.batch_id] - dn * self.bs)[:, 0, :], GB2a),
                                         (self.DX[self.batch_id] - dn * self.bs)[:, 0, :])
        grad_beta = grad_beta1 + grad_beta2

        return 0.5 * np.concatenate([np.sum(grad_alpha, axis=0), np.sum(grad_beta, axis=0)]) * 1 / len(self.batch_id)

    # compute the hessian of the quasi-lik at point par for a given sde object

    def hessian(self, param, batch_id=None, **kwargs):

        assert self.sde.model.mode == 'sym', 'Gradient computation available only in symbolic mode'

        try:
            self.update_aux(param, batch_id)
        except np.linalg.LinAlgError:
            return np.diag(np.full(shape=len(self.sde.model.param), fill_value=np.finfo(float).resolution))

        dn = self.sde.sampling.delta

        Jbs = np.moveaxis(self.sde.model.der_foo["Jb"](*self.X[self.batch_id].transpose(), **param), -1, 0)
        # DSs = np.array([self.model.der_foo["DS"](*x, **param) for x in self.X[:-1]])
        DSs = np.moveaxis(self.sde.model.der_foo["DS"](*self.X[self.batch_id].transpose(), **param), -1, 0)

        # Hbs = np.array([self.model.der_foo["Hb"](*x, **param) for x in self.X[:-1]])
        # HSs = np.array([self.model.der_foo["HS"](*x, **param) for x in self.X[:-1]])
        Hbs = np.moveaxis(self.sde.model.der_foo["Hb"](*self.X[self.batch_id].transpose(), **param), -1, 0)
        # DSs = np.array([self.model.der_foo["DS"](*x, **param) for x in self.X[:-1]])
        HSs = np.moveaxis(self.sde.model.der_foo["HS"](*self.X[self.batch_id].transpose(), **param), -1, 0)

        HESSA1 = np.einsum('nd, ndpq -> npq', self.DXS_inv[:, 0, :], Hbs)
        HESSA2 = np.einsum('npd, ndq -> npq', Jbs.transpose((0, 2, 1)), np.einsum('nde, nep -> ndp', self.Ss_inv, Jbs))
        hess_alpha = np.sum(-2 * HESSA1 + 2 * HESSA2, axis=0)

        GB1 = np.einsum('nde, npef -> npdf', self.Ss_inv, DSs)
        GB2a = np.einsum('npde, nef -> npdf', GB1, self.Ss_inv)

        HESSB1 = np.trace(np.einsum('npde, nqef -> npqdf', GB1, GB1), axis1=3, axis2=4)
        HESSB2 = np.trace(np.einsum('nde, npqef -> npqdf', self.Ss_inv, HSs), axis1=3, axis2=4)

        HESSB3a = np.einsum('npde, nef, nqfg -> npqdg', DSs, self.Ss_inv, DSs)
        HESSB3b = np.einsum('nde, npqef, nfg -> npqdg', self.Ss_inv, HSs, self.Ss_inv)
        HESSB3c = HESSB3a - HESSB3b + HESSB3a.transpose((0, 2, 1, 3, 4))

        HESSB3 = np.einsum('nd, npqde, ne -> npq', self.DXS_inv[:, 0, :], HESSB3c, self.DXS_inv[:, 0, :])

        hess_beta = - HESSB1 + HESSB2 - 1 / dn * HESSB3
        # make sure result is symmetric
        hess_beta = np.sum(0.5 * (hess_beta + hess_beta.transpose((0, 2, 1))), axis=0)
        hess_ab = -2* np.sum(np.einsum('nd, npde, neq -> npq', (self.DX[self.batch_id] - dn * self.bs)[:, 0, :], GB2a, Jbs), axis=0)

        hess = 0.5*np.block([[hess_alpha, hess_ab.transpose()], [hess_ab, hess_beta]]) * 1 / len(self.batch_id)
        return hess

