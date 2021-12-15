from .sdelearn import Sde
import numpy as np


class SdeLearner:
    def __init__(self, sde):

        # contains SDE
        self.sde = sde
        # contains estimate
        self.est = None
        # contains var cov matrix
        self.vcov = None
        # contains info about optimization
        self.optim_info = {'args': None}

    def set_sde(self, sde):
        # resets the object with new sde (possibly needed in penalized estimation when subclass of base est is not explicitely
        # known
        self.__init__(sde)
        return

    def fit(self, **kwargs):
        """
        fit the model in sde, update the estimate ana vcov value. The fit method is expected to catch its args
        in optim_info. This is used in validation of penalized estimation
        :param kwargs:
        """
        # remember to catch args in subclass implementation of this method!!
        self.optim_info['args'] = {**kwargs}
        # remember to check the order of the supplied values and bounds!
        # the following is a template of what should be at the beginning of the fit methods
        start = {} # hypotetical dict of starting values
        bounds = []  # hypotetical list of bounds, assumed to have same order as start
        if kwargs.get('bounds') is not None:
            bounds = kwargs['bounds']
            ik = [list(start.keys()).index(k) for k in self.sde.model.param]
            bounds = [bounds[i] for i in ik]
            kwargs['bounds'] = bounds

        start = {k: start.get(k) for k in self.sde.model.param}


        pass

    def predict(self, sampling=None, x0=None, n_rep=1000, **kwargs):
        """
        montecarlo estimate of average value
        :param x0: starting point for simulations, if None the first observation of data is used
        :param n_rep: number of simulations for estimating expected value
        :param sampling: SdeSampling object specifying where predictions take place. If note same as learner's sde is used
        :param kwargs: params passed to specific methods
        :return: dataframe with predictions
        """

        if sampling is None:
           new_samp = self.sde.sampling
        else:
            new_samp = sampling

        if x0 is None:
            x0 = self.sde.data.original_data[0]

        new_sde = Sde(model=self.sde.model, sampling=new_samp, data=None)
        pred_data = new_sde.simulate(self.est, x0, ret_data=True)/n_rep
        for i in range(n_rep - 1):
            pred_data += new_sde.simulate(self.est, x0, ret_data=True) / n_rep

        new_sde.set_data(pred_data).data.format_data(time_index=new_sde.sampling.grid, col_names=new_sde.model.state_var)

        return new_sde.data.data

    def loss(self, param):
        """
        loss function for the learner (e.g. qmle, lsa ...) at point param
        :param param: parameter value
        :return:
        """
        # make sure to sort the param dictionary to have the same order as self.sde.model.param
        # in case the computation is based on position in the dict and not on name (e.g. if the dict is converted to np.array)
        pass

    def gradient(self, param):
        """
        gradient of the loss function at point param
        :param param: parameter value
        :return:
        """
        pass

    def hessian(self, param, **kwargs):
        """
        hessian of the loss function at point param
        :param **kwargs:
        :param param: parameter value
        :param kwargs:
        :return:
        """
        pass

    @staticmethod
    def generate_batch(batch, sde_learn, **kwargs):
        return np.random.choice(a=sde_learn.sde.data.n_obs-1, size=int(np.floor(batch * sde_learn.sde.data.n_obs)),
                                replace=False)

    @staticmethod
    def loss_wrap(par, sde_learn, **kwargs):
        """
        wrapping function for optimizers
        :param par:
        :param sde_learn:
        :return:
        """
        param = dict(zip(sde_learn.sde.model.param, par))
        return sde_learn.loss(param=param, **kwargs)

    @staticmethod
    def grad_wrap(par, sde_learn, **kwargs):
        """
       wrapping function for optimizers
       :param par:
       :param sde_learn:
       :return:
       """
        param = dict(zip(sde_learn.sde.model.param, par))
        return sde_learn.gradient(param=param, **kwargs)

    @staticmethod
    def hess_wrap(par, sde_learn, **kwargs):
        param = dict(zip(sde_learn.sde.model.param, par))
        return sde_learn.hessian(param, **kwargs)

    @staticmethod
    def backtrack_rule(f, x, jac_x, alpha=0.5, gamma=0.8, **kwargs):
        s = 1
        if f(x - s * jac_x, **kwargs) <= f(x, **kwargs) - alpha * s * np.linalg.norm(jac_x) ** 2:
            return s

        while f(x - s * jac_x, **kwargs) > f(x, **kwargs) - alpha * s * np.linalg.norm(jac_x) ** 2:
            s = gamma * s

        return s

    # accelerated gradient descent. Supports projections into a BOX only, passed as list of pairs (lower, upper)
    # x0 is an np array, f and jac are assumed to take in input np.arrays, jac returns np.array or list, converted to np.array
    @staticmethod
    def nesterov_descent(f, x0, jac=None, epsilon=1e-03, max_it=1e3, bounds=None, cyclic=False, batch=None, **kwargs):

        x_prev = np.array(x0)
        t_prev = 1
        y_prev = x_prev
        x_curr = x_prev
        y_curr = y_prev

        if bounds is not None:
            bounds = np.array(bounds).transpose()
            assert np.all(x_prev > bounds[0]) and np.any(x_prev < bounds[1]), 'starting point outside of bounds'

        if batch is not None:
            batch_id = SdeLearner.generate_batch(batch, **kwargs)
        else:
            batch_id = None

        jac_y = np.array(jac(y_prev, batch_id=batch_id, **kwargs))
        padding = np.ones_like(jac_y)

        if cyclic:
            padding = np.zeros_like(jac_y)
            cycle_start = np.argmax(jac_y)
            padding[cycle_start] = 1

        it_count = 1
        status = 1
        message = ''

        while np.linalg.norm(jac_y) >= epsilon and it_count < max_it:

            if batch is not None:
                batch_id = SdeLearner.generate_batch(batch, **kwargs)

            s = SdeLearner.backtrack_rule(f, y_prev, jac_y * padding, alpha=0.5, gamma=0.8, batch_id=batch_id, **kwargs)
            # print('s ' + str(s) + '\n')

            x_curr = y_prev - s * jac_y * padding

            if bounds is not None:
                x_curr[x_curr < bounds[0]] = bounds[0][x_curr < bounds[0]] + epsilon
                x_curr[x_curr > bounds[1]] = bounds[1][x_curr > bounds[1]] - epsilon

            # # interrupt execution before costly evaluation of the gradient
            # if np.linalg.norm(x_curr - x_prev) < 0.1 * epsilon * np.linalg.norm(x_prev):
            #     message = 'Relative reduction less than {0}'.format(0.1*epsilon)
            #     return {'x': x_curr, 'f': f(x_curr, **kwargs), 'status': status, 'message': message, 'niter': it_count, 'jac': jac_y, 'epsilon': epsilon}

            t_curr = (1 + np.sqrt(1 + 4 * t_prev ** 2)) / 2

            y_curr = x_curr + (t_prev - 1) / t_curr * (x_curr - x_prev)

            if bounds is not None:
                y_curr[y_curr < bounds[0]] = bounds[0][y_curr < bounds[0]] + epsilon
                y_curr[y_curr > bounds[1]] = bounds[1][y_curr > bounds[1]] - epsilon

            x_prev = x_curr
            t_prev = t_curr
            y_prev = y_curr
            #
            jac_y = np.array(jac(y_curr, batch_id=batch_id, **kwargs))

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

        if np.linalg.norm(jac_y) >= epsilon and it_count >= max_it:
            message = 'Maximum number of iterations reached'
        if np.linalg.norm(jac_y) < epsilon:
            message = 'Success: gradient norm less than epsilon'
            status = 0
        if batch is not None:
            jac_y = np.array(jac(y_curr, batch_id=None, **kwargs))

        return {'x': x_curr, 'f': f(x_curr, batch_id=None, **kwargs), 'status': status, 'message': message, 'niter': it_count,
                'jac': jac_y, 'epsilon': epsilon, 'batch': batch}

    def __str__(self):
        out = 'SDE Learner: ' + type(self).__name__
        return out