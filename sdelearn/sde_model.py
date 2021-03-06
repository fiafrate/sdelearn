import numpy as np
import sympy as sym


class SdeModel:
    def __init__(self, drift, diff, mod_shape=None, par_names=None, state_var=None, mode='sym'):

        '''
                if mode is not "fun" (meaning mode="sym", the default expected behavior)
                 either parameter names or var names should be inferred from expressions
        '''
        self.mode = mode
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
        # if there are contants in the expression the result will be a single scalar,
        # resulting in a ragged array. Operations on such arrays appear to be deprecated in numpy
        # TEMP SOLVED: ADDED INVISIBLE MULTIPLICATION BY ZERO SYMBOL TO CONSTANT EXPRESSIONS
        self.der_expr = None
        self.der_foo = None

        if par_names is not None:
            self.drift_par = par_names["drift"]
            self.diff_par = par_names["diffusion"]
            self.param = self.drift_par + self.diff_par

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
            sym_dr_names = [s.name for expr in self.b_expr for s in expr.free_symbols]
            # var_names = [s.name for s in state_var]
            self.drift_par = list(set(sym_dr_names) - set(self.state_var))
            self.drift_par.sort()

            sym_di_names = [s.name for row in self.A_expr for expr in row for s in expr.free_symbols]
            self.diff_par = list(set(sym_di_names) - set(self.state_var))
            self.diff_par.sort()

            self.param = self.drift_par + self.diff_par

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

        self.npar_dr = len(self.drift_par)
        self.npar_di = len(self.diff_par)

        self.par_groups = {'drift': self.drift_par, 'diff': self.diff_par}

    def set_param(self, par_names):
        self.drift_par = par_names["drift"]
        self.diff_par = par_names["diffusion"]
        self.param = [self.drift_par, self.diff_par]
        self.par_groups = {'drift': self.drift_par, 'diff': self.diff_par}

    def set_var_names(self, var_names):
        self.state_var = var_names

    def set_der(self):
        Jb_expr = np.array(
            [[expr.diff(param) + self.null_expr
              for param in self.drift_par] for expr in self.b_expr])
        Hb_expr = np.array(
            [[[expr.diff(param1, param2) + self.null_expr
               for param1 in self.drift_par] for param2 in self.drift_par] for expr in self.b_expr])

        DS_expr = np.array([[[expr.diff(param) + self.null_expr
                              for expr in row] for row in self.S_expr] for param in self.diff_par])
        HS_expr = np.array(
            [[[[expr.diff(param1, param2) + self.null_expr
                for expr in row] for row in self.S_expr] for param2 in self.diff_par] for param1 in self.diff_par])

        # we are applying numpy operations to expressions! this seems to work...

        #
        # C_expr = np.matmul(DA_expr, self.A_expr.T)
        # DS_expr = C_expr + C_expr.transpose((0, 2, 1))
        #
        # D_expr = np.matmul(HA_expr, self.A_expr.T)
        # E_expr = np.swapaxes(np.dot(DA_expr, DA_expr.transpose((0, 2, 1))), 1, 2)
        # HS_expr = D_expr + D_expr.transpose((0, 1, 3, 2)) + E_expr + E_expr.transpose((0, 1, 3, 2))

        der_expr = {"b": sym.Matrix(self.b_expr), "Jb": sym.Array(Jb_expr), "Hb": sym.Array(Hb_expr),
                    "S": sym.Matrix(self.S_expr), "DS": sym.Array(DS_expr), "HS": sym.Array(HS_expr)}
        return der_expr

    def __str__(self):
        out = '\nSde Model object ---- \n\nNumber of Gaussian noises: {0}\nNumber of state variables: {1}' \
              '\nNumber of parameters: {2}'.format(self.n_noise, self.n_var, len(self.param))
        return out
