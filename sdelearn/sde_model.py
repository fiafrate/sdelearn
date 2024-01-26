import warnings

import numpy as np
import sympy as sym

from sympy.tensor.array import derive_by_array

class SdeModel:
    """
    - Represents a model for stochastic differential equations (SDEs) with symbolic capabilities.
    - Handles drift and diffusion components of the SDE.
    - Supports both symbolic (default) and functional modes.

    """
    def __init__(self, drift, diff, mod_shape=None, par_names=None, state_var=None, mode='sym', options={}):
        """
        :param dift: either symbolic expression, if mode == 'sym', or function, if mode == 'fun', modelling the drift
        :param diff: either symbolic expression, if mode == 'sym', or function, if mode == 'fun', modelling the diffusion
        :param par_names: dictionary with keys 'drift' and 'diffusion' containing list of characters naming parameters,
                to be supplied in functional mode, ignored in symbolic mode
        :param mod_shape: tuple (n_var, n_noise), to be supplied only in fun mode
        :param state_var: list of variables names in the equations. If None, x0, ..., x_nvar will be taken.
            It should not be None in symbolic mode
        :param mode: either 'fun' or 'sym'
        :param options: dictionary with keys
            - 'hess', boolean values, controls if second derivatives are computed (in 'sym' mode)
        """
        self.mode = mode

        self.options = {'hess': False}
        if options is not None:
            o_l = list(self.options.keys())
            for k, v in options.items():
                if k in o_l:
                    self.options[k] = v
                else:
                    warnings.warn('unknown Model option ' + k)

        if mode == 'sym':
            self.n_var = len(drift)
            self.n_noise = len(diff[0])
        else:
            self.n_var = mod_shape[0]
            self.n_noise = mod_shape[1]

        self.drift_par = None
        self.diff_par = None
        self.param = None

        self.state_var = state_var

        # dictionaries containing derivatives of drift and diffusion
        # der_expr contains expressions, der_foo contains "lambdified" functions,
        # as a function of both x and param. These functions are NOT properly
        # vectorized as a result of https://github.com/sympy/sympy/issues/5642:
        # if there are constants in the expression the result will be a single scalar,
        # resulting in a ragged array. Operations on such arrays appear to be deprecated in numpy
        # TEMP SOLVED: ADDED INVISIBLE MULTIPLICATION BY ZERO SYMBOL TO CONSTANT EXPRESSIONS
        self.der_expr = None
        self.der_foo = None

        if par_names is not None:
            self.set_param(par_names)

        if state_var is None:
            self.state_var = ["x{0}".format(x) for x in range(self.n_var)]

        # set drift and diffusion: if mode="fun" directly set as the input, otherwise
        # lambdify the expressions and then set drift and diffusion

        if mode == 'sym':
            self.null_expr = sym.Mul(sym.symbols(self.state_var[0]), 0, evaluate=False)

            # convert list expressions into numpy arrays
            self.b_expr = np.array([v + self.null_expr for v in drift])
            self.A_expr = np.array([[v + self.null_expr for v in row] for row in diff])
            self.S_expr = self.A_expr @ self.A_expr.transpose()
            self.S_expr = np.array([[v + self.null_expr for v in row] for row in self.S_expr])

            # infer parameter names
            par_names = {}

            sym_dr_names = [s.name for expr in self.b_expr for s in expr.free_symbols]
            par_names['drift'] = list(set(sym_dr_names) - set(self.state_var))

            sym_di_names = [s.name for row in self.A_expr for expr in row for s in expr.free_symbols]
            par_names['diffusion'] = list(set(sym_di_names) - set(self.state_var))

            self.set_param(par_names)


            # create auxiliary functions for evaluating drift and diff expressions

            # b and A take in input arrays
            self.b = sym.lambdify(self.state_var + self.param, sym.Array(self.b_expr))
            self.A = sym.lambdify(self.state_var + self.param, sym.Matrix(self.A_expr))

            def drift_wrap(x, param):
                return np.array(self.b(*x, **param))

            def diff_wrap(x, param):
                return np.array(self.A(*x, **param))

            # take in input array (or list) and dictionary for parameters
            # this ensures compatibility with mode='fun'
            self.drift = drift_wrap
            self.diff = diff_wrap

            # compute symbolic derivatives of drift and diffusion
            self.der_expr = self.set_der()
            self.der_foo = {k: sym.lambdify(self.state_var + self.param, v, 'numpy') for k, v in self.der_expr.items()}

        if mode == 'fun':
            self.drift = drift
            self.diff = diff


    def set_param(self, par_names):
        '''
        setup parameter info: store list of params, groups and numbers. Instantiate corresponding fields
        :param par_names: dictionary with keys 'drift' and 'diffusion' containing list of characters naming parameters
        :return:
        '''

        self.par_groups = {}

        if par_names.get('drift') is None or len(par_names.get('drift')) == 0:
            self.drift_par = []
        else:
            self.drift_par = [v for v in par_names["drift"]]
            self.drift_par.sort()
            self.par_groups['drift'] = self.drift_par

        if par_names.get('diffusion') is None or len(par_names.get('diffusion')) == 0:
            self.diff_par = []
        else:
            self.diff_par = [v for v in par_names["diffusion"]]
            self.diff_par.sort()
            self.par_groups['diff'] = self.diff_par

        self.param = self.drift_par + self.diff_par
        self.npar_dr = len(self.drift_par)
        self.npar_di = len(self.diff_par)



    def set_var_names(self, var_names):
        self.state_var = var_names

    def set_der(self):
        if len(self.drift_par) > 0:

            # Jb_expr = np.array(
            #     [[expr.diff(param) + self.null_expr
            #       for param in self.drift_par] for expr in self.b_expr])
            Jb_expr = derive_by_array(self.b_expr, sym.symbols(self.drift_par)).transpose()
            n0 = np.full(Jb_expr.shape, self.null_expr)
            Jb_expr = Jb_expr + n0

            if self.options.get('hess') or self.options.get('hess') is None:
                Hb_expr = np.array(
                    [[[expr.diff(param1, param2) + self.null_expr
                       for param1 in self.drift_par] for param2 in self.drift_par] for expr in self.b_expr])
            else:
                Hb_expr = [[[0]]]
        else:
            Jb_expr = [[0]]
            Hb_expr = [[[0]]]

        if len(self.diff_par) > 0:
            # DS_expr = np.array([[[expr.diff(param) + self.null_expr
            #                       for expr in row] for row in self.S_expr] for param in self.diff_par])
            DA_expr = derive_by_array(self.A_expr, sym.symbols(self.diff_par))
            n0 = np.full(DA_expr.shape, self.null_expr)
            DA_expr = DA_expr + n0

            if self.options.get('hess') or self.options.get('hess') is None:
                HA_expr = derive_by_array(DA_expr, sym.symbols(self.diff_par))
                n0 = np.full(HA_expr.shape, self.null_expr)
                HA_expr = HA_expr + n0
            else:
                HA_expr = [[[[0]]]]
                # HS_expr = np.array(
            #     [[[[expr.diff(param1, param2) + self.null_expr
            #         for expr in row] for row in self.S_expr] for param2 in self.diff_par] for param1 in self.diff_par])
        else:
            # DS_expr = [[[0]]]
            # HS_expr = [[[[0]]]]
            DA_expr = [[[0]]]
            HA_expr = [[[[0]]]]
        #
        # C_expr = np.matmul(DA_expr, self.A_expr.T)
        # DS_expr = C_expr + C_expr.transpose((0, 2, 1))
        #
        # D_expr = np.matmul(HA_expr, self.A_expr.T)
        # E_expr = np.swapaxes(np.dot(DA_expr, DA_expr.transpose((0, 2, 1))), 1, 2)
        # HS_expr = D_expr + D_expr.transpose((0, 1, 3, 2)) + E_expr + E_expr.transpose((0, 1, 3, 2))

        # der_expr = {"b": sym.Matrix(self.b_expr), "Jb": sym.Array(Jb_expr), "Hb": sym.Array(Hb_expr),
        #             "S": sym.Matrix(self.S_expr), "DS": sym.Array(DS_expr), "HS": sym.Array(HS_expr)}
        der_expr = {"b": sym.Matrix(self.b_expr), "Jb": sym.Array(Jb_expr), "Hb": sym.Array(Hb_expr),
                    "A": sym.Matrix(self.A_expr), "DA": sym.Array(DA_expr), "HA": sym.Array(HA_expr)}
        return der_expr

    def __str__(self):
        out = '\nSde Model object ---- \n\nNumber of Gaussian noises: {0}\nNumber of state variables: {1}' \
              '\nNumber of parameters: {2}'.format(self.n_noise, self.n_var, len(self.param))
        return out
