import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

from .sdelearn import Sde
from .sde_qmle import Qmle
from .sde_learner import SdeLearner
import warnings

import copy


class AdaElasticNet(SdeLearner):
    def __init__(self, sde, base_estimator, lsa=True, weights=None, q=1, delta=0, alpha=1, penalty=None, n_pen=100, adaptive=True, hess_check=False, **kwargs):
        """
        Adaptive Elastic Net estimator.
        :param sde: a Sde object
        :param base_estimator: a base estimator of class SdeLearner or a string naming a subclass for constructing one (e.g. Qmle)
        :param lsa: if True uses least squares approximation to compute the lasso solution. Otherwise a penalty term is added to the loss
            of base_est.
        :param weights: adaptive weights/penalties for each parameter, in the form of a dictionary `{param: value}`,
            defaults to None, in that case all weights are set to 1
        :param q: exponent of the lq penalty of the bridge estimator. Either a scalar or an array with same length
            as the parameter groups in Sde. Currently only q=1 is supported
        :param delta: adjusts adaptive penalties as w_i / |est_i|^delta, defaults to zero meaning the adaptive weights
        given in argument `weights` remain unchanged
        :param alpha: in (0, 1], controls the mixing of elastic net penalty. alpha=1 corresponds to lasso,
            alpha ->0 tends  to ridge-like estimator, alpha=0 is not supported.
        :param penalty: grid of lambda values at which to evaluate the solution path, defaults to None meaning that
            100 log-spaced values will be used from 0 to lambda_max
        :param n_pen: number of penalty values to consider (counting 0, ignored if penalty is supplied)
        :param hess_check: make sure hessian matrix is spd
        :**kwargs: arguments to be passed to fit method of base estimator if not already fitted
        """
        super().__init__(sde=sde)

        self.q = q
        self.alpha = alpha
        self.base_est = None
        self.ini_est = None
        self.ini_hess = None
        self.lsa = lsa
        self.adaptive = adaptive



        # instantiate base est if not already present
        if isinstance(base_estimator, SdeLearner):
            self.base_est = base_estimator
        else:
            if base_estimator == 'Qmle':
                self.base_est = Qmle(sde)


        if self.lsa:
            self.adaptive = True



        if self.adaptive:
            # LSA MODE: fit base estimator if not already fitted
            if self.base_est.est is None:
                self.base_est.fit(**kwargs)
                self.ini_est = self.base_est.est
            else:
                self.ini_est = self.base_est.est

            ini_arr = np.array(list(self.ini_est.values()))

            # extract hessian matrix at max point
            self.ini_hess = self.base_est.optim_info['hess']
            self.ini_hess = 0.5 * (self.ini_hess + self.ini_hess.T)
            if hess_check:
                # fix hessian matrix if not symmetric positive definite
                v, a = sp.linalg.eigh(self.ini_hess)
                # set to zero negative eigs + some small noise (closest SPD approx)
                # v[v < 0] = 0.001 * np.abs(np.random.randn(len(v[v < 0])))
                # replace neg eigvals  with positive vales
                v[v < 0] = np.abs(v[v < 0])
                self.ini_hess = a @ np.diag(v) @ a.transpose()
                # lipschitz constant of quadratic part
                self.lip = np.max(v)
            else:
                self.lip = sp.linalg.eigvalsh(self.ini_hess,
                                              subset_by_index=[self.ini_hess.shape[0]-1, self.ini_hess.shape[0]-1])[0]




            # info about parameter groups, used in block estimate

            # names in initial est
            self.ini_names = list(self.ini_est.keys())
            # indices of names per group
            self.group_idx = {k: [self.ini_names.index(par) for par in v] for k, v in self.sde.model.par_groups.items()}
            self.group_names = list(self.group_idx.keys())
            # setup block lipschitiz contants
            block_hess = [self.ini_hess[self.group_idx.get(k)][:, self.group_idx.get(k)] for k in self.group_idx.keys()]
            self.block_lip = np.array([np.linalg.eigvalsh(bh).max() for bh in block_hess])

            # setup weights
            self.delta = delta
            self.weights = weights if weights is not None else dict(zip(sde.model.param, [1] * len(sde.model.param)))

            self.w_ada = np.array(list(self.weights.values())) / \
                         np.power(np.abs(ini_arr), delta)

        else:
            # if not adaptive store param names and groups from sde object
            self.ini_names = self.sde.model.param
            self.group_idx = {k: [self.ini_names.index(par) for par in v] for k, v in self.sde.model.par_groups.items()}
            self.group_names = list(self.group_idx.keys())
            self.weights = weights if weights is not None else dict(zip(sde.model.param, [1] * len(sde.model.param)))
            # not truly adaptive, just keep the name for compatibility
            self.w_ada = np.array(list(self.weights.values()))


        # maximal lambda value (starting point for backward algorithms: self.proximal_gradient(0, lambda_max) = 0
        #c_q = (2 * (1 - q)) ** (1 / (2 - q)) * (1 + 0.5 * q / (1 - q))


        if self.lsa:
            grad_0 = self.ini_hess @ np.array(list(self.ini_est.values()))
        else:
            grad_0 = self.base_est.grad_wrap(np.zeros(len(self.ini_names)), self.base_est)

        self.lambda_max = np.max(np.abs(grad_0) / (self.alpha * self.w_ada))

        self.lambda_ini = None
        # if self.adaptive:
        #     self.lambda_ini = 0.5 * np.min(np.abs(ini_arr / self.w_ada)) * self.lip
        # else:
        #     self.lambda_ini = 0.0001 * self.lambda_max
        st_pen = np.min([n_pen**(-2) * self.lambda_max, 0.00001])
        # optimal lambda value, computed after cross validation ("lambda.1se")
        self.lambda_opt = None
        # lambda corresponding to min cv score ("lambda.min")
        self.lambda_min = None

        if penalty is None:
            self.n_pen = n_pen
            self.penalty = np.zeros(n_pen)
            self.penalty[1:] = np.exp(
                np.linspace(start=np.log(st_pen), stop=np.log(self.lambda_max), num=n_pen - 1))
            self.penalty[n_pen - 1] = self.lambda_max
        else:
            self.n_pen = len(penalty)
            self.penalty = np.sort(penalty)

        # initialize solution path
        self.est_path = np.empty((len(self.penalty), len(self.ini_names)))
        if self.adaptive:
            self.est_path[0] = ini_arr
        # last value set as zero -- try to estimate backwards
        self.est_path[self.n_pen - 1] = np.zeros(len(self.ini_names))
        # details on optim_info results
        self.path_info = [None] * (self.n_pen - 2)

        # info on CV
        self.val_loss = None

    def loss(self, param, penalty=1, **kwargs):
        """
        Adaptive Elastic Net loss function, using either exact base est loss or its quadratic approximation
        :param param: parameter value at which loss is computed (supports dictionary or array, assuming that the order is consistent)
        :param penalty: constant penalty multiplying the adaptive weights
        :param kwargs: additional arguments to be passed to base estimator loss fun
        :return: scalar loss value

        """
        if isinstance(param, dict):
            par = np.array(list(param.values()))
        else:
            par = param
        w_a = self.w_ada

        reg = penalty * (self.alpha * np.linalg.norm(par * w_a,ord= self.q) ** self.q
                         + 0.5 * (1 - self.alpha) * np.linalg.norm(par * w_a,ord=2)**2)
        if self.lsa:
            par_ini = np.array(list(self.ini_est.values()))
            out = 0.5 * (par - par_ini) @ self.ini_hess @ (par - par_ini) + reg
        else:
            out = self.base_est.loss_wrap(par, self.base_est, **kwargs) + reg


        return out


    def gradient(self, param, **kwargs):
        """
        Gradient of the *unpenalized* loss used to build a penalized estimator, either exact (e.g. quasi-likelihood)
        or least square approximation, depending on how the learner was built. N.B. this does not coincide with the gradient
        of self.loss (which is not differentiable). This is primarily used in backtracking.

        :param param: value at which to compute gradient, either dict or np.array (with parameters assumed to be in
            the same order initial estimate)
        :param kwargs: additional arguments to be passed to base estimator loss
        :return: numpy array of gradient vector
        """

        if isinstance(param, dict):
            y = np.array(list(param.values()))
        else:
            y = param

        if self.lsa:
            par_ini = np.array(list(self.ini_est.values()))
            jac_y = self.ini_hess @ (y - par_ini)
        else:
            jac_y = self.base_est.grad_wrap(y, self.base_est, **kwargs)

        return jac_y

    def fit(self, cv=None, nfolds=5, cv_metric="loss", aic_k=1, **kwargs):
        """
        :param cv: controls validation of the lambda parameter in the lasso path. Possible values are:
            - None no validation takes place.
            - cv in (0,1): proportion of obs to be used as validation. E.g. if cv = 0.1 the last 10% of
        obs is used as validation and the first 90% as training.
            - cv = True: k-fold cross validation.
        Performance on validation set is measured using the loss
        function of the base estimator. Finally the estimate path on the full dataset is computed: the field est
        is filled with the estimate corresponding  to the optimal lambda, which is in turn saved in the field lambda_opt
        :param nfolds: number of folds for cross validation (considered only if cv = True)
        :param cv_metric: is cv is not None controls the loss metric to evaluate on validation sets, either
            "loss" (e.g. quasi likelihood) or "mse" (prediction mse, based on Monte Carlo predictions as in SdeLearner.predict)
        :param aic_k: k value to be used in AIC computation, ie -loglik + k * nonzero par
        :param kwargs: additional arguments for controlling optimization. See description of `proximal_gradient`
        :return: self
        """

        self.optim_info['args'] = {'cv': cv, 'nfolds': nfolds, "cv_metric": cv_metric, **kwargs}
        # in this case start is initial estimate, assumed to be already in model order
        # and the bounds are assumed to be in the same order, so no check on the order is needed

        if cv is None:
            for i in range(len(self.est_path) - 2):
                # compute est
                cur_est = self.proximal_gradient(self.est_path[self.n_pen - 1 - i], self.penalty[self.n_pen - 2 - i],
                                                 **kwargs)
                # store results
                self.path_info[self.n_pen - 3 - i] = cur_est
                self.est_path[self.n_pen - 2 - i] = cur_est['x']

            self.val_loss = self.aic(k=aic_k)

            # compute lamba_opt using iqr instead std
            iqr = np.quantile(self.val_loss, 0.5, method='midpoint') - np.quantile(self.val_loss, 0.0, method='midpoint')
            self.lambda_opt = self.penalty[1:-1][self.val_loss < np.nanmin(self.val_loss) + iqr][-1]
            self.lambda_min = self.penalty[np.nanargmin(self.val_loss) + 1]

        elif 0 < cv < 1:
            n = self.sde.data.n_obs
            n_val = int(cv * n)

            # create auxiliary sde objects
            sde_tr = copy.deepcopy(self.sde)
            sde_val = copy.deepcopy(self.sde)
            # sde_val = Sde(model=self.sde.model, sampling=self.sde.sampling.sub_sampling(last_n=n_val))
            # sde_tr = Sde(model=self.sde.model, sampling=self.sde.sampling.sub_sampling(first_n=n - n_val + 1))

            sde_tr.sampling = self.sde.sampling.sub_sampling(first_n=n - n_val + 1)
            sde_val.sampling = self.sde.sampling.sub_sampling(last_n=n_val)
            sde_val.set_data(self.sde.data.data.iloc[-n_val:])
            sde_tr.set_data(self.sde.data.data.iloc[:n - n_val + 1])

            # create auxiliary estimator of same type of base est, on training data
            aux_est = type(self.base_est)(sde_tr)
            aux_est.fit(start=self.ini_est, **self.base_est.optim_info['args'])
            enet_tr = AdaElasticNet(sde_tr, base_estimator=aux_est, weights=self.weights, delta=self.delta,
                                     alpha=self.alpha, penalty=self.penalty)
            enet_tr.fit(**kwargs)

            # create aux est object on validation data and compute loss
            aux_est = type(self.base_est)(sde_val)
            val_loss = np.full(len(enet_tr.est_path) - 2, np.nan)
            for i in range(len(enet_tr.est_path) - 2):
                try:
                    if cv_metric == "loss":
                        val_loss[i] = aux_est.loss(dict(zip(aux_est.sde.model.param, enet_tr.est_path[i+1]))) #+ np.sum(enet_tr.est_path[i+1] != 0)/sde_val.data.n_obs
                    elif cv_metric == "mse":
                        n_rep = kwargs.get('n_rep') if kwargs.get('n_rep') is not None else 100
                        enet_tr.est = dict(zip(aux_est.sde.model.param, enet_tr.est_path[i+1]))
                        val_loss[i] = \
                            np.nanmean((enet_tr.predict(sampling=sde_val.sampling, x0=sde_val.data.data.iloc[0].to_numpy(),
                                                      n_rep=n_rep).to_numpy() - sde_val.data.data.to_numpy()) ** 2)
                except:
                    pass

            val_loss[np.isinf(val_loss)] = np.nan

            # compute full path
            self.fit(**kwargs)

            # compute final estimate using optimal lambda
            self.lambda_opt = self.penalty[1:-1][val_loss < np.nanmin(val_loss) + np.nanstd(val_loss)][-1]
            self.lambda_min = self.penalty[np.nanargmin(val_loss)+1]
            self.val_loss = val_loss

        elif cv == True:
            n = self.sde.data.n_obs
            ntr0 = int(0.4 * n)
            nval = n - ntr0
            nkth = int(nval / nfolds)

            sde_tr = copy.deepcopy(self.sde)
            sde_val = copy.deepcopy(self.sde)
            # array to store loss values
            val_loss = np.full((self.n_pen - 2, nfolds), np.nan)
            for k in range(nfolds):
                # create auxiliary sde objects
                sde_tr.sampling = self.sde.sampling.sub_sampling(from_range=[0, ntr0 + k * nkth])
                sde_tr.set_data(self.sde.data.data.iloc[0:(ntr0 + k * nkth)])
                if k < nfolds - 1:
                    sde_val.sampling = self.sde.sampling.sub_sampling(
                        from_range=[ntr0 + k * nkth - 1, ntr0 + (k + 1) * nkth])
                    sde_val.set_data(self.sde.data.data.iloc[(ntr0 + k * nkth - 1):(ntr0 + (k + 1) * nkth)])
                else:
                    sde_val.sampling = self.sde.sampling.sub_sampling(from_range=[ntr0 + k * nkth - 1, n])
                    sde_val.set_data(self.sde.data.data.iloc[(ntr0 + k * nkth - 1):])

                # create auxiliary estimator of same type of base est, on training data
                aux_est = type(self.base_est)(sde_tr)
                aux_est.fit(start=self.ini_est, **self.base_est.optim_info['args'])
                enet_tr = AdaElasticNet(sde_tr, base_estimator=aux_est, weights=self.weights, delta=self.delta,
                                         alpha=self.alpha, penalty=self.penalty)
                enet_tr.fit(**kwargs)

                # create aux est object on validation data and compute loss
                aux_est = type(self.base_est)(sde_val)

                for i in range(len(enet_tr.est_path) - 2):
                    try:
                        if cv_metric == "loss":
                            val_loss[i, k] = aux_est.loss(dict(zip(aux_est.sde.model.param, enet_tr.est_path[i+1]))) #+ np.sum(enet_tr.est_path[i+1] != 0)/sde_val.data.n_obs
                        elif cv_metric == "mse":
                            enet_tr.est = dict(zip(aux_est.sde.model.param, enet_tr.est_path[i+1]))
                            n_rep = kwargs.get('n_rep') if kwargs.get('n_rep') is not None else 100
                            pred = enet_tr.predict(sampling=sde_val.sampling, x0=sde_val.data.data.iloc[0].to_numpy(),
                                                    n_rep=n_rep).to_numpy()
                            val_loss[i, k] = np.mean((pred - sde_val.data.data.to_numpy()) ** 2)
                    except:
                        pass

            # cv loss
            val_loss[np.isinf(val_loss)] = np.nan
            cv_loss = np.nanmean(val_loss, axis=1)
            cv_std = np.nanstd(val_loss, axis=1)/np.sqrt(nfolds)
            # compute full path
            self.fit(**kwargs)

            # compute final estimate using optimal lambda
            self.lambda_opt = self.penalty[1:-1][cv_loss < np.nanmin(cv_loss) + cv_std][-1]
            self.lambda_min = self.penalty[np.nanargmin(cv_loss)]
            self.val_loss = val_loss

        self.est = self.coef(self.lambda_min)
        self.vcov = np.linalg.inv(self.ini_hess)
        return self

    def soft_threshold(self, par, penalty, padding=None):
        """
        enet soft-thresholding operator
        :param q: order of thresholding
        :param par: evaluation point
        :param penalty: penalty parameter multiplying adaptive weights in Sde.AdaLasso object
        :param padding: optionally compute soft thresh on a subvector indexed by padding == 1
        :return: vector of component wise soft-thresholding of par
        """
        if padding is None:
            padding = np.ones_like(par)
        w = self.w_ada[padding == 1]

        out = np.zeros_like(par)
        i_nz = np.abs(par) > penalty * self.alpha * w
        out[i_nz] = np.sign(par[i_nz]) * (np.abs(par[i_nz]) - penalty * self.alpha * w[i_nz]) / (1 + penalty * (1 - self.alpha))
        return out


    def prox_backtrack(self, y_curr, penalty, gamma=0.8, s_ini=1, **kwargs):

        s = s_ini
        max_ls = 50  # max line search attempts

        jac_y = self.gradient(y_curr)
        x_curr = self.soft_threshold(y_curr - s * jac_y, penalty * s)
        x_par = dict(zip(self.sde.model.param, x_curr))
        y_par = dict(zip(self.sde.model.param, y_curr))

        f_s = self.loss(x_par, penalty=0, **kwargs)
        q_s = self.loss(y_par, penalty=0, **kwargs) + (x_curr - y_curr) @ jac_y \
              + (0.5 / s) * np.linalg.norm(x_curr - y_curr, ord=2) ** 2
        if f_s <= q_s:
            return s
        k = 1

        while f_s > q_s and k < max_ls:
            s *= gamma
            x_curr = self.soft_threshold(y_curr - s * jac_y, penalty * s)
            x_par = dict(zip(self.sde.model.param, x_curr))
            f_s = self.loss(x_par, penalty=0, **kwargs)
            q_s = self.loss(y_par, penalty=0, **kwargs) + (x_curr - y_curr) @ jac_y \
                  + (0.5 / s) * np.linalg.norm(x_curr - y_curr, ord=2) ** 2
            k += 1

        return s

    def proximal_gradient(self, x0, penalty, epsilon=1e-03, max_it=1000, bounds=None,
                          opt_alg="fista",
                          backtracking=False,
                          s_ini = 10,
                          verbose=False,
                          **kwargs):
        """
        compute proximal gradient algorithms for regularization path optimization.
        :param x0: array-like starting point
        :param penalty: value penalty parameter
        :param epsilon: relative convergence factor: converge if ||x_prev - x_curr|| < 0.5 * epsilon / Lip_const
        :param max_it: maximum number of iterations allowed
        :param bounds: matrix-like, parameter bounds with rows [par_min, par_max]
        :param opt_alg: choose optimization algorithm: either "mAPG" (monotone APG as in Li, Lin 2015), "cyclic" (coordinate-wise with custom order) or "block_wise"
            if sde model has block paramters
        :param backtracking: boolean, if true use backtracking to determine stepsize, otherwise use relevant lip constant
        :param s_ini: initial stepsize value
        :param kwargs
        :return: dict 'x': last x point reached, 'f': loss function at x, 'status': convergence status 0/1,
            'message': convergence message, 'niter': number of iterations,\
                'jac': gradient of f at x, 'epsilon': epsilon}
        """
        if self.lsa:
            par_ini = np.array(list(self.ini_est.values()))

        it_count = 1

        x_prev = np.array(x0)
        y_curr = np.copy(x_prev)
        z_curr = np.copy(x_prev)

        t_prev = 0
        t_curr = 1

        # fix bounds
        if bounds is not None:
            bounds = bounds.transpose()
            y_curr[y_curr < bounds[0]] = bounds[0][y_curr < bounds[0]] + epsilon
            y_curr[y_curr > bounds[1]] = bounds[1][y_curr > bounds[1]] - epsilon

        padding = np.ones_like(x0)

        assert opt_alg in ["fista", "cyclic", "block_wise"], 'invalid opt_alg'
        assert not (opt_alg == 'block_wise' and len(self.group_names) == 1), 'invalid opt_alg'

        block_end = True

        # initialize optimization algorithms
        if opt_alg == 'fista':
            if self.adaptive:
                s_lip = 0.9 / self.lip
        elif opt_alg == "cyclic":
            padding = np.zeros_like(x_prev)
            if self.lsa:
                jac_y = self.ini_hess @ (x_prev - par_ini)
            else:
                jac_y = self.base_est.grad_wrap(x_prev, self.base_est, **kwargs)
            cycle_start = np.argmax(jac_y)
            padding[cycle_start] = 1
            block_end = False
            if self.adaptive:
                s_lip = 1 / np.diag(self.ini_hess)[padding == 1]

        elif opt_alg == "block_wise":
            padding = np.zeros_like(x_prev, dtype=int)
            padding[self.group_idx[self.group_names[(it_count - 1) % len(self.group_names)]]] = 1
            block_end = False
            if self.adaptive:
                s_lip = 0.9 / self.block_lip[(it_count - 1) % len(self.group_names)]

        # stepsize choice
        if backtracking:
            s = self.prox_backtrack(y_curr, penalty=penalty, gamma=0.8, s_ini=s_ini)
            # s = max(s, s_lip)
        else:
            s = s_lip

        # compute full vector of updates even in coordinate/block case. Otherwise, a
        # single coordinate (or a block) doesn't change algorithm would stop, even if convergence is not reached!
        # x_soft = self.soft_threshold(y_curr - s * jac_y, penalty * s)
        # x_curr = np.where(padding == 1, x_soft, x_prev)

        if self.lsa:
            jac_y = self.ini_hess[padding == 1, :] @ (x_prev - par_ini)
        else:
            jac_y = self.base_est.grad_wrap(x_prev, self.base_est, **kwargs)[padding == 1]

        x_curr = np.copy(x_prev)
        x_curr[padding == 1] = self.soft_threshold(x_prev[padding == 1] - s * jac_y, penalty * s, padding)

        while (np.linalg.norm(x_curr - x_prev, ord=2) >= epsilon or not block_end) and it_count < max_it:

            if verbose:
                if it_count % 100 == 1:
                    print(str(np.where(self.penalty == penalty)) + '/' + str(self.n_pen) + ': ' + str(it_count))
                    print('norm: ' + str(np.linalg.norm(x_curr - x_prev, ord=2)))
                    print('stepsize: ' + str(s))
            it_count += 1

            if opt_alg == 'fista':

                if backtracking:
                    s = self.prox_backtrack(y_curr, penalty=penalty, s_ini=s)
                    # print(s)
                    # s = max(s, s_lip)

                # t_curr = (1 + np.sqrt(1 + 4 * t_prev ** 2)) / 2
                #
                # y_curr = x_curr + (t_prev - 1) / t_curr * (x_curr - x_prev)
                #
                # #
                #
                # if self.lsa:
                #     jac_y = self.ini_hess @ (y_curr - par_ini)
                # else:
                #     jac_y = self.base_est.grad_wrap(y_curr, self.base_est, **kwargs)
                #
                # x_prev = np.copy(x_curr)
                # x_soft = self.soft_threshold(y_curr - s * jac_y, penalty * s)
                # x_curr = np.where(padding == 1, x_soft, x_prev)
                #
                # t_prev = t_curr
                t_prev = t_curr
                t_curr = (1 + np.sqrt(1 + 4 * t_prev ** 2)) / 2
                y_curr = x_curr + (t_prev - 1) / t_curr * (x_curr - x_prev)

                # MONOTONE VARIANT -------------------------------------------
                # y_curr = x_curr + t_prev / t_curr * (z_curr - x_curr) + (t_prev - 1) / t_curr * (x_curr - x_prev)
                # jac_y = self.gradient(y_curr)
                # jac_x = self.gradient(x_curr)
                #
                # z_curr = self.soft_threshold(par=y_curr - s * jac_y, penalty=penalty * s)
                # v_curr = self.soft_threshold(par=x_curr - s * jac_x, penalty=penalty * s)
                #
                # x_prev = np.copy(x_curr)
                # if self.loss(z_curr) <= self.loss(v_curr):
                #     x_curr = z_curr
                # else:
                #     x_curr = v_curr
                #

                x_prev = np.copy(x_curr)
                jac_y = self.gradient(y_curr, **kwargs)
                x_curr = self.soft_threshold(par=y_curr - s * jac_y, penalty=penalty * s)


            elif opt_alg == "cyclic":
                # auxiliary vector y_curr is updated coordinate-wise.
                # x_prev serves as a monitor to be checked only when the full vector has been updated
                padding = np.roll(padding, 1)
                block_end = False
                if np.argmax(padding) == cycle_start:
                    # at beginning of new cycle
                    block_end = True
                    jac_y = self.gradient(y_curr, **kwargs)
                    cycle_start = np.argmax(jac_y)
                    padding = np.zeros_like(x_prev)
                    padding[cycle_start] = 1
                    x_prev = np.copy(x_curr)

                elif np.argmax(np.roll(padding, 1)) == cycle_start:
                    # at the end of current cycle
                    block_end = True

                if backtracking:
                    s = self.prox_backtrack(y_curr, penalty=penalty, s_ini=s)
                else:
                    s = 1 / np.diag(self.ini_hess)[padding == 1]

                y_curr = np.copy(x_curr)
                if self.lsa:
                    jac_y = self.ini_hess[padding == 1, :] @ (y_curr - par_ini)
                else:
                    jac_y = self.base_est.grad_wrap(y_curr, self.base_est, **kwargs)[padding == 1]
                x_curr[padding == 1] = self.soft_threshold(y_curr[padding == 1] - s * jac_y, penalty * s, padding)

            elif opt_alg == "block_wise":
                padding = np.zeros_like(x_prev, dtype=int)
                padding[self.group_idx[self.group_names[(it_count - 1) % len(self.group_names)]]] = 1
                block_end = False

                if it_count % len(self.group_names) == 0:
                    # at end of current cycle can check condition
                    block_end = True

                elif it_count % len(self.group_names) == 1:
                    # at beginning of new cycle store prev values
                    x_prev = np.copy(x_curr)
                #
                if backtracking:
                    s = self.prox_backtrack(y_curr, penalty=penalty, s_ini=s)
                else:
                    s = 0.9 / self.block_lip[(it_count - 1) % len(self.group_names)]
                y_curr = np.copy(x_curr)

                if self.lsa:
                    jac_y = self.ini_hess[padding == 1, :] @ (y_curr - par_ini)
                else:
                    jac_y = self.base_est.grad_wrap(y_curr, self.base_est, **kwargs)[padding == 1]

                x_curr[padding == 1] = self.soft_threshold(y_curr[padding == 1] - s * jac_y, penalty * s, padding)

            # fix bounds
            if bounds is not None:
                x_curr[x_curr < bounds[0]] = bounds[0][x_curr < bounds[0]] + epsilon
                x_curr[x_curr > bounds[1]] = bounds[1][x_curr > bounds[1]] - epsilon

        if np.linalg.norm(x_curr - x_prev, ord=2) >= epsilon:
            message = 'Maximum number of iterations reached'
            status = 1
        else:
            message = 'Success: increment norm smaller than threshold'
            status = 0

        return {'x': x_curr, 'f': self.loss_wrap(x_curr, self), 'status': status, 'message': message, 'niter': it_count,
                'jac': jac_y, 'epsilon': epsilon, 'stepsize': s}

    def plot(self, save_fig=None):
        plt.figure()
        plt.title('Coefficients path')
        plt.ylabel('Estimates')
        plt.xlabel('log lambda')
        plt.plot(np.log(self.penalty), self.est_path)
        if save_fig is None:
            plt.show()
        else:
            plt.savefig(save_fig)
        return self

    def coef(self, penalty):
        return dict(zip(self.sde.model.param, self.est_path[np.where(self.penalty == penalty)[0][0]]))

    def aic(self, k=1):
        nzp = np.sum(self.est_path[1:-1] != 0, axis=1) / self.sde.data.n_obs
        loss = [self.base_est.loss(self.coef(pen)) for pen in self.penalty[1:-1]]
        aic = loss + k * nzp
        return aic