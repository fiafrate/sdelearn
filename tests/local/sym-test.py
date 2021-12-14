import cmath
import math
import numpy as np
import sympy as sym
import string

from sympy.utilities.autowrap import ufuncify

y, z = sym.symbols('y z')

f = sym.cos(z * y)
f.subs([(z, 2)]).evalf()

math.cos(6)
f_prime = sym.diff(f, z)
sym.diff(f, "z", "y")
f_second = sym.diff(f_prime, y)

f_second.subs([(z, 1), (z, 2)])

f_foo = sym.lambdify(['y', 'z'], f, 'numpy')
val = np.vstack((np.arange(10), np.full(10, math.pi)))
np.round(f_foo(np.arange(10) / 2, math.pi))

f_dict = {"f": f, "f_prime": f_prime}
f_lam = {k: sym.lambdify(['y', 'z'], v, 'numpy') for k, v in f_dict.items()}
f_lam["f"](np.arange(10), math.pi)

theta_dr = [sym.symbols('theta_dr{0}{1}'.format(i, j)) for i in range(2) for j in range(2)]
theta_di = [sym.symbols('theta_di{0}{1}'.format(i, j)) for i in range(2) for j in range(2)]

# theta_dr = [sym.symbols('theta{0}'.format(i)) for i in range(2)]
# theta_di = [sym.symbols('theta2')]

all_param = theta_dr + theta_di
state_var = [sym.symbols('x{0}'.format(i)) for i in range(2)]
# state_var = [sym.symbols('x{0}'.format(i)) for i in range(1)]

b_expr = np.array([theta_dr[0] - theta_dr[1] * state_var[0], theta_dr[2] - theta_dr[3] * state_var[1]])
b_expr = [theta_dr[0] - theta_dr[1] * state_var[0], theta_dr[2] - theta_dr[3] * state_var[1]]
# b_expr = np.array([theta_dr[0] - theta_dr[1] * state_var[0]])
b_expr = sym.Matrix([theta_dr[0] - theta_dr[1] * state_var[0], theta_dr[2] - theta_dr[3] * state_var[1]])

A_expr = np.array(
    [[theta_di[0] - theta_di[1] * state_var[0], sym.sympify('0')],
     [sym.sympify('0'), theta_di[2] - theta_di[3] * state_var[1]]])
A_expr = sym.Matrix(A_expr)
A_expr = sym.Matrix(
    [[theta_di[0] - theta_di[1] * state_var[0], sym.sympify('0')],
     [sym.sympify('0'), theta_di[2] - theta_di[3] * state_var[1]]])
A_expr = [[theta_di[0] - theta_di[1] * state_var[0], sym.sympify('0')],
          [sym.sympify('0'), theta_di[2] - theta_di[3] * state_var[1]]]


null_expr = sym.Mul(state_var[0], 0, evaluate=False)

A_expr = np.array([[v + null_expr if v.is_constant() else v for v in row] for row in A_expr])


# A_expr = np.array([[theta_di[0] * state_var[0]]])
truep = {"theta_dr00": 0, "theta_dr01": +0.5, "theta_dr10": 0, "theta_dr11": +0.5, "theta_di00": +1,
         "theta_di01": 0, "theta_di10": +1, "theta_di11": 1}

# truep = {"theta0": 0.25, "theta1": +0.5, "theta2": 2}

b_expr[0].evalf(subs=truep)
A_expr[0][0].evalf(subs=truep)

# list(b_expr[0].free_symbols)[0].name
sym_dr_names = [s.name for expr in b_expr for s in expr.free_symbols]
var_names = [s.name for s in state_var]
par_dr_names = list(set(sym_dr_names) - set(var_names))
par_dr_names.sort()

sym_di_names = [s.name for row in A_expr for expr in row for s in expr.free_symbols]
par_di_names = list(set(sym_di_names) - set(var_names))
par_di_names.sort()

par_names = par_dr_names + par_di_names

npar_dr = len(par_dr_names)
npar_di = len(par_di_names)

b_foo = sym.lambdify(var_names + par_names, b_expr, 'numpy')
A_foo = sym.lambdify(state_var + all_param, A_expr, 'numpy')
A_foo = ufuncify(tuple(state_var + all_param), A_expr[0, 1])

Xa = np.array(np.arange(6).reshape(3, 2))
asd = np.array([A_foo(*x, **truep) for x in Xa])
asd = np.asarray(A_foo(*Xa.transpose(), **truep))

A_foo(*Xa.transpose(), **truep)
A_foo(*Xa[1], **truep)

np.swapaxes(A_foo(*Xa.transpose(), **truep), 0, -1)
asd = np.empty([len(Xa)] + list(A_expr.shape), dtype=float)
asd = np.array( )

[[1, 2], [3, 4]]
# def eval_np_sym(np_sym, x, param):
#     f"""
#     :param np_sym: numpy array of sympy expressions to be evaluated
#     :param x: *dictionary* of x values of the form var_name[i] = value[i] at which to evaluate
#     :param param: *dictionary* of param values at which to evaluate
#     :return: np array of numeric evaluations, with same shape as np_sym
#     """
#     arr_eval = np.empty_like(np_sym, dtype=float)
#     sym_eval = {**x, **param}
#     with np.nditer(arr_eval, flags=['refs_ok', 'multi_index'], op_flags=['writeonly']) as it:
#         for u in it:
#             u[...] = np_sym[it.multi_index].evalf(subs=sym_eval)
#     return arr_eval
def eval_np_sym(np_sym, x):
    f"""
    :param np_sym: numpy array of sympy expressions to be evaluated
    :param x: *dictionary* of x values of the form var_name[i] = value[i] at which to evaluate
    :param param: *dictionary* of param values at which to evaluate
    :return: np array of numeric evaluations, with same shape as np_sym
    """
    arr_eval = np.empty_like(np_sym, dtype=float)
    sym_eval = {**x, **param}
    with np.nditer(arr_eval, flags=['refs_ok', 'multi_index'], op_flags=['writeonly']) as it:
        for u in it:
            u[...] = np_sym[it.multi_index].evalf(subs=sym_eval)
    return arr_eval


def sub_np_sym(np_sym, x):
    f"""
    :param np_sym: numpy array of sympy expressions to be substituted
    :param x: list of x values to substitute
    :return: np array of substitutions evaluations, with same shape as np_sym
    """
    arr_eval = np.empty_like(np_sym)
    sym_subs = [(u, v) for u, v in zip(state_var, x)]
    with np.nditer(arr_eval, flags=['multi_index', 'refs_ok'], op_flags=['writeonly']) as it:
        for u in it:
            u[...] = np_sym[it.multi_index].subs(sym_subs)
    return arr_eval


def b(x, param):
    x_ev = dict(zip(var_names, x))
    return eval_np_sym(np_sym=b_expr, x=x_ev, param=param)


b([1, 2], truep)


def A(x, param):
    x_ev = dict(zip(var_names, x))
    return eval_np_sym(np_sym=A_expr, x=x_ev, param=param)


A([1, 2], truep)

Jb_expr = np.array([[expr.diff(param) if not expr.diff(param).is_constant() else expr.diff(param) + null_expr for param in par_dr_names] for expr in b_expr])
Hb_expr = np.array(
    [[[expr.diff(param1, param2)  if not expr.diff(param1, param2).is_constant() else expr.diff(param1, param2) + null_expr for param1 in par_dr_names] for param2 in par_dr_names] for expr in b_expr])

# DA_expr = np.array([[[expr.diff(param) for expr in row] for row in A_expr] for param in par_di_names])
# HA_expr = np.array(
#     [[[[expr.diff(param1, param2) for expr in row] for row in A_expr] for param2 in par_di_names] for param1 in
#      par_di_names])

S_expr = A_expr @ A_expr.transpose()
S_expr = np.array([[v + null_expr if v.is_constant() else v for v in row] for row in S_expr])

DS_expr = np.array([[[expr.diff(param) if not expr.diff(param).is_constant() else expr.diff(param) + null_expr for expr in row] for row in S_expr] for param in par_di_names])
HS_expr = np.array(
    [[[[expr.diff(param1, param2) if not expr.diff(param1, param2).is_constant() else expr.diff(param1, param2) + null_expr for expr in row] for row in S_expr] for param2 in par_di_names] for param1 in
     par_di_names])

# C_expr = np.matmul(DA_expr, A_expr.T)
# DS_expr = C_expr + C_expr.transpose((0, 2, 1))
#
# D_expr = np.matmul(HA_expr, A_expr.T)
# E_expr = np.swapaxes(np.dot(DA_expr, DA_expr.transpose((0, 2, 1))), 1, 2)
# HS_expr = D_expr + D_expr.transpose((0, 1, 3, 2)) + E_expr + E_expr.transpose(((0, 1, 3, 2)))
# np.all(HS_expr == HS_expr1)

def set_aux_expr():
    Jb_expr = np.array([[expr.diff(param) for param in par_dr_names] for expr in b_expr])
    Hb_expr = np.array(
        [[[expr.diff(param1, param2) for param1 in par_dr_names] for param2 in par_dr_names] for expr in b_expr])

    DA_expr = np.array([[[expr.diff(param) for expr in row] for row in A_expr] for param in par_di_names])
    HA_expr = np.array(
        [[[[expr.diff(param1, param2) for expr in row] for row in A_expr] for param2 in par_di_names] for param1 in
         par_di_names])

    S_expr = A_expr @ A_expr.transpose()

    C_expr = np.matmul(DA_expr, A_expr.T)
    DS_expr = C_expr + C_expr.transpose((0, 2, 1))

    D_expr = np.matmul(HA_expr, A_expr.T)
    E_expr = np.swapaxes(np.dot(DA_expr, DA_expr.transpose((0, 2, 1))), 1, 2)
    HS_expr = D_expr + D_expr.transpose((0, 1, 3, 2)) + E_expr + E_expr.transpose(((0, 1, 3, 2)))

    b_expr2 = b_expr
    aux_expr = {"b": b_expr2, "Jb": Jb_expr, "Hb": Hb_expr, "DA": DA_expr, "HA": HA_expr, "S": S_expr, "DS": DS_expr,
                "HS": HS_expr}
    return aux_expr


def set_aux_f():
    Ss = np.empty((X.shape[0] - 1, X.shape[1], X.shape[1]), dtype=object)
    bs = np.empty((X.shape[0] - 1, 1, X.shape[1]), dtype=object)
    Jbs = np.empty((X.shape[0] - 1, X.shape[1], npar_dr), dtype=object)
    DSs = np.empty((X.shape[0] - 1, npar_di, X.shape[1], X.shape[1]), dtype=object)
    Hbs = np.empty((X.shape[0] - 1, X.shape[1], npar_dr, npar_dr), dtype=object)
    HSs = np.empty((X.shape[0] - 1, npar_di, npar_di, X.shape[1], X.shape[1]), dtype=object)


aux_expr = set_aux_expr()

#n_obs = sde.data.data.shape[0]
n_obs = 10
#n_var = sde.data.data.shape[1]
n_var = 2

# X = sym.symarray('x', (n_obs, n_var))
X = np.round(np.random.randn(n_obs * n_var).reshape(n_obs, n_var) * 10, 2)
# X = np.array([[0.7],[1.3]])
X = np.array(sde.data.data)
Xr = X.reshape(n_obs, 1, n_var)
DX = Xr[1:len(Xr)] - Xr[:len(Xr) - 1]

# dn = sde.sampling.delta
dn = 0.1

#
# S_mat  = sym.Array(S_expr)
# b_mat  = sym.Array(b_expr)
# Jb_mat = sym.Array(Jb_expr)
# DS_mat = sym.Array(DS_expr)
# Hb_mat = sym.Array(Hb_expr)
# HS_mat = sym.Array(HS_expr)


HS_arr = sym.Array(HS_expr)
HS_arr.subs([('x0', 1), ('x1', 2)])

Sf = np.empty((X.shape[0] - 1, X.shape[1], X.shape[1]), dtype=object)
bf = np.empty((X.shape[0] - 1, 1, X.shape[1]), dtype=object)
Jbf = np.empty((X.shape[0] - 1, X.shape[1], npar_dr), dtype=object)
DSf = np.empty((X.shape[0] - 1, npar_di, X.shape[1], X.shape[1]), dtype=object)
Hbf = np.empty((X.shape[0] - 1, X.shape[1], npar_dr, npar_dr), dtype=object)
HSf = np.empty((X.shape[0] - 1, npar_di, npar_di, X.shape[1], X.shape[1]), dtype=object)

#
# for i in range(len(X)-1):
#     Sf[i]  = S_mat.subs([(u,v) for u,v in zip(state_var, X[i])])
#     bf[i]  = b_mat.subs([(u,v) for u,v in zip(state_var, X[i])])
#     Jbf[i] = Jb_mat.subs([(u,v) for u,v in zip(state_var, X[i])])
#     DSf[i] = DS_mat.subs([(u,v) for u,v in zip(state_var, X[i])])
#     Hbf[i] = Hb_mat.subs([(u,v) for u,v in zip(state_var, X[i])])
#     HSf[i] = HS_mat.subs([(u,v) for u,v in zip(state_var, X[i])])


for i in range(len(X) - 1):
    bf[i] = sub_np_sym(b_expr, X[i])
    Sf[i] = sub_np_sym(S_expr, X[i])
    Jbf[i] = sub_np_sym(Jb_expr, X[i])
    DSf[i] = sub_np_sym(DS_expr, X[i])
    Hbf[i] = sub_np_sym(Hb_expr, X[i])
    HSf[i] = sub_np_sym(HS_expr, X[i])

bf = sym.lambdify(all_param, bf, "numpy")
Sf = sym.lambdify(all_param, Sf, "numpy")
Jbf = sym.lambdify(all_param, Jbf, "numpy")
DSf = sym.lambdify(all_param, DSf, "numpy")
Hbf = sym.lambdify(all_param, Hbf, "numpy")
HSf = sym.lambdify(all_param, HSf, "numpy")

bf(**truep)

# direclty lambdify expressions? -----------------------------------------------
param = truep

param = qmle.est
bf = sym.lambdify(var_names + all_param, b_expr, "numpy")
Sf = sym.lambdify(var_names + all_param, S_expr, "numpy")
Jbf = sym.lambdify(var_names + all_param, Jb_expr, "numpy")
DSf = sym.lambdify(var_names + all_param, DS_expr, "numpy")
Hbf = sym.lambdify(var_names + all_param, Hb_expr, "numpy")
HSf = sym.lambdify(var_names + all_param, HS_expr, "numpy")

Ss = np.array([Sf(*x, **param) for x in X[:-1]])
bs = np.array([bf(*x, **param) for x in X[:-1]]).reshape((X.shape[0] - 1, 1, X.shape[1]))
Jbs = np.array([Jbf(*x, **param) for x in X[:-1]])
DSs = np.array([DSf(*x, **param) for x in X[:-1]])
Hbs = np.array([Hbf(*x, **param) for x in X[:-1]])
HSs = np.array([HSf(*x, **param) for x in X[:-1]])

Ss_inv = np.linalg.inv(Ss)
DXbs = DX - dn * bs
DXS_inv = np.matmul(DXbs, Ss_inv)
asd = np.matmul(DXS_inv, DX.transpose(0, 2, 1)).squeeze()
asd = DX.transpose(0, 2, 1)
# --------------------------------------------------------------------


# direclty lambdify AND vectorize expressions? -----------------------------------------------
param = truep

param = qmle.est
bf = sym.lambdify(var_names + all_param, sym.Matrix(b_expr), "numpy")
Sf = sym.lambdify(var_names + all_param, sym.Matrix(S_expr), "numpy")
Jbf = sym.lambdify(var_names + all_param, sym.Array(Jb_expr), "numpy")
DSf = sym.lambdify(var_names + all_param, sym.Array(DS_expr), "numpy")
Hbf = sym.lambdify(var_names + all_param, sym.Array(Hb_expr), "numpy")
HSf = sym.lambdify(var_names + all_param, sym.Array(HS_expr), "numpy")


Ss2 = np.swapaxes(Sf(*X[:-1].transpose(), **param), 0, -1)
bs2 = np.swapaxes(bf(*X[:-1].transpose(), **param), 0, -1)
Jbs2 = np.moveaxis(Jbf(*X[:-1].transpose(), **param), -1, 0)
Hbs2 = np.moveaxis(Hbf(*X[:-1].transpose(), **param), -1, 0)
DSs2 = np.moveaxis(DSf(*X[:-1].transpose(), **param), -1, 0)
HSs2 = np.moveaxis(HSf(*X[:-1].transpose(), **param), -1, 0)

#
#
# Ss2 = np.array([Sf(*x, **param) for x in X[:-1]])
# bs2 = np.array([bf(*x, **param) for x in X[:-1]]).transpose(0, 2, 1)
# Jbs2 = np.array([Jbf(*x, **param) for x in X[:-1]])
# DSs2 = np.array([DSf(*x, **param) for x in X[:-1]])
# Hbs2 = np.array([Hbf(*x, **param) for x in X[:-1]])
# HSs2 = np.array([HSf(*x, **param) for x in X[:-1]])

np.all(Ss2 == Ss)
np.all(bs2 == bs)
np.all(Jbs2 == Jbs)
np.all(DSs2 == DSs)
np.all(Hbs2 == Hbs)
np.all(HSs2 == HSs)


Ss_inv2 = np.linalg.inv(Ss2)
DXbs2 = DX - dn * bs
DXS_inv2 = np.matmul(DXbs, Ss_inv)
asd = np.matmul(DXS_inv, DX.transpose(0, 2, 1)).squeeze()
asd = DX.transpose(0, 2, 1)



# Ss = np.empty((X.shape[0], X.shape[1], X.shape[1]), dtype=float)
# bs = np.empty((X.shape[0], 1, X.shape[1]), dtype=float)
# Jbs = np.empty((X.shape[0], X.shape[1], npar_dr), dtype=float)
# DSs = np.empty((X.shape[0], npar_di, X.shape[1], X.shape[1]), dtype=float)
# Hbs = np.empty((X.shape[0], X.shape[1], npar_dr, npar_dr), dtype=float)
# HSs = np.empty((X.shape[0], npar_di, npar_di, X.shape[1], X.shape[1]), dtype=float)

param = truep
param = qmle.est
X_ev = X.transpose()
asd = Sf(*np.array([[1, 2], [3, 4]]), **param)
Ss = np.array(Sf(*X_ev, **param))
bs = np.array(bf(*X, **param))
Jbs = np.array(Jbf(**param))
DSs = np.array(DSf(**param))
Hbs = np.array(Hbf(**param))
HSs = np.array(HSf(**param))

Ss_inv = np.linalg.inv(Ss)
DXS_inv = np.matmul(DX - dn * bs, Ss_inv)
#
# A_mat = sym.Matrix(A_expr)
# A_mat = sym.Array(A_expr)
#
# A_mat.subs([(u,v) for u,v in zip(state_var, X[0])])
#
# DA_mat = sym.Array(DA_expr)
# DA_mat.subs([(u,v) for u,v in zip(state_var, X[0])])


1 / (truep['theta2'] ** 2 * X[0] ** 2) * (X[1] - X[0] - dn * (truep['theta0'] - truep['theta1'] * X[0]))

1 / (truep['theta2'] ** 2 * X[0]) * (X[1] - X[0] - dn * (truep['theta0'] - truep['theta1'] * X[0]))

- 1 / truep['theta2'] + 1 / (dn * truep['theta2'] ** 3 * X[0] ** 2) * (
            X[1] - X[0] - dn * (truep['theta0'] - truep['theta1'] * X[0])) ** 2
