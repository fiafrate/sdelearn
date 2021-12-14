import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sde_sampling import SdeSampling as Samp
from sde_model import SdeModel as Mod
from sde_data import SdeData as Data


class Sde:
    def __init__(self, sampling, model, data=None):
        # this assumes inputs are of suitable classes, or None for data
        self.model = model
        self.sampling = sampling  # modify grid according to data?
        self.data = data

        if data is not None:
            assert self.model.n_var == self.data.data.shape[1], 'Number of variables in model does not match the shape of data!'
            assert self.sampling.n == self.data.n_obs, 'Number of observation in sampling does not match the shape of data!'
        # add checks on input!

    def set_sampling(self, initial, terminal, n=None, delta=None):
        self.sampling = Samp(initial, terminal, n, delta)
        return self

    def set_model(self, drift, diff, mod_shape, par_names=None, var_names=None):
        self.model = Mod(drift, diff, mod_shape, par_names, var_names)
        return self

    def set_data(self, data):
        self.data = Data(data)
        assert self.model.n_var == self.data.data.shape[1], 'Number of variables in model does not match the shape of data!'
        assert self.sampling.n == self.data.n_obs, 'Number of observation in sampling does not match the shape of data!'
        return self

    def simulate(self, truep, x0, ret_data=False):
        # x0 can be None, in that case the value in sampling will be used
        # optionally returns data, useful in predictions
        # should check that model and sampling is not None

        # assign parameter names if missing
        self.model.param = list(truep.keys())

        sim = np.empty([self.sampling.n, self.model.n_var])
        if self.sampling.x0 is None:
            self.sampling.x0 = x0
        sim[0] = x0
        for i in range(1, np.shape(sim)[0]):
            cur_drift = np.array(self.model.drift(sim[i - 1], truep))
            cur_diff = np.array(self.model.diff(sim[i - 1], truep))

            sim[i] = sim[i - 1] + cur_drift * self.sampling.delta \
                     + np.dot(cur_diff, np.sqrt(self.sampling.delta) * np.random.normal(size=self.model.n_noise))

        # either directly return data (useful in predictions) or set the current object and return it
        if ret_data:
            return sim
        else:
            self.set_data(sim)
            self.data.format_data(time_index=self.sampling.grid, col_names=self.model.state_var)

        return self

    def plot(self):
        #plt.figure()
        # plt.plot(self.sampling.grid, self.data.data)
        self.data.data.plot()
        return self

    def __str__(self):
        return '\nSde with components:\n'+self.model.__str__() + '\n' + self.sampling.__str__()
