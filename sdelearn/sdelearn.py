import numpy as np

from .sde_sampling import SdeSampling as Samp
from .sde_model import SdeModel as Mod
from .sde_data import SdeData as Data


class Sde:
    """
    Central class that integrates the components of SDEs: sampling, model, and data.
    Enables operations like setting sampling parameters, defining the model, and simulating data based on the model.
    Built-in checks ensure consistency and compatibility across components.

    Attributes

    - `model`: An instance of the SdeModel class representing the SDE model.
    - `sampling`: An instance of the SdeSampling class representing the sampling scheme.
    - `data`: An optional instance of the SdeData class representing the observed time series.

    """
    def __init__(self, sampling, model, data=None):
        # this assumes inputs are of suitable classes, or None for data
        self.model = model
        self.sampling = sampling  # modify grid according to data?
        self.data = data
        if data is not None:
            assert self.model.n_var == self.data.data.shape[1], 'Number of variables in model does not match the shape of data!'
            assert self.sampling.n == self.data.n_obs, 'Number of observation in sampling does not match the shape of data!'
            self.data.format_data(time_index=self.sampling.grid, col_names=self.model.state_var)
            self.sampling.x0 = np.array(self.data.data.iloc[0])

        # add checks on input!

    def set_sampling(self, initial, terminal, n=None, delta=None):
        self.sampling = Samp(initial, terminal, n, delta)
        return self

    def set_model(self, drift, diff, mod_shape, par_names=None, var_names=None):
        self.model = Mod(drift, diff, mod_shape, par_names, var_names)
        return self

    def set_data(self, data, format_data=True):
        """
        sets the data attribute of Sde to data. If format = True (the default) data is formatted to have timestamp
        and names corresponding to sampling and model attributes.
        :param data: array-like object containing data, supported by pandas for data frame creation
        :return: self
        """
        self.data = Data(data)
        assert self.model.n_var == self.data.data.shape[1], 'Number of variables in model does not match the shape of data!'
        assert self.sampling.n == self.data.n_obs, 'Number of observation in sampling does not match the shape of data!'
        if format_data:
            self.data.format_data(time_index=self.sampling.grid, col_names=self.model.state_var)
        self.sampling.x0 = np.array(self.data.data.iloc[0])

        return self

    def simulate(self, param, x0, bounds=None, ret_data=False):
        """

        :param param: parameter value to be used in the simulation
        :param x0: starting point, if None the value in sde.sampling will be used
        :param bounds: bounds for the state space. List of pairs (lower, upper) for every state variable.
        :param ret_data: boolean, either return the simulated data or set sde.data.
        :return: simulated data, if ret_data, else self.
        """
        # optionally returns data, useful in predictions
        # should check that model and sampling is not None

        # assign parameter names if missing
        if self.model.param is None:
            self.model.param = list(param.keys())

        if bounds is not None:
            bounds = bounds.transpose()
            tol = np.sqrt(self.sampling.delta)

        # fix parameter order to match self params
        param = {k: param.get(k) for k in self.model.param}
        sim = np.empty([self.sampling.n, self.model.n_var])
        if self.sampling.x0 is None:
            self.sampling.x0 = x0
        sim[0] = x0

        for i in range(1, np.shape(sim)[0]):
            cur_drift = np.array(self.model.drift(sim[i - 1], param))
            cur_diff = np.array(self.model.diff(sim[i - 1], param))

            sim[i] = sim[i - 1] + cur_drift * self.sampling.delta \
                     + np.dot(cur_diff, np.sqrt(self.sampling.delta) * np.random.normal(size=self.model.n_noise))

            if bounds is not None:
                sim[i][sim[i] < bounds[0]] = bounds[0][sim[i] < bounds[0]] + 0.1 * (bounds[0][sim[i] < bounds[0]] - sim[i][sim[i] < bounds[0]])
                sim[i][sim[i] > bounds[1]] = bounds[1][sim[i] > bounds[1]] - 0.1 * (sim[i][sim[i] > bounds[1]] - bounds[1][sim[i] > bounds[1]])

        # either directly return data (useful in predictions) or set the current object and return it
        if ret_data:
            return sim
        else:
            self.set_data(sim)


        return self

    def plot(self, save_fig=None, **kwargs):
        """
        :param save_fig: either None (default) in which case figure is shown, or string naming the file
        the plot will be saved in
        :param kwargs: optional graphical parameters, see pandas plot for reference
        :return: self
        """
        if save_fig is None:
            self.data.data.plot(**kwargs)
        else:
            self.data.data.plot(**kwargs).figure.savefig(save_fig)
        return self

    def __str__(self):
        return '\nSde with components:\n'+self.model.__str__() + '\n' + self.sampling.__str__()
