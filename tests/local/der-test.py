import numpy as np

from sdelearn import *
from sde_data import *
from sde_sampling import *
from sde_model import *
from sde_qmle import *


def b(x, param):
    out = [0, 0]
    out[0] = param["theta.dr00"] - param["theta.dr01"] * x[0]
    out[1] = param["theta.dr10"] - param["theta.dr11"] * x[1]
    return out


def A(x, param):
    out = [[0, 0], [0, 0]]
    out[0][0] = param["theta.di00"] + param["theta.di01"] * x[0]
    out[1][1] = param["theta.di10"] + param["theta.di11"] * x[1]
    out[1][0] = 0
    out[0][1] = 0
    return out


sde = Sde(sampling=SdeSampling(initial=0, terminal=2, delta=0.01),
          model=SdeModel(b, A, mod_shape=[2, 2],
                         par_names={"drift": ["theta.dr00", "theta.dr01", "theta.dr10", "theta.dr11"],
                                    "diffusion": ["theta.di00", "theta.di01", "theta.di10", "theta.di11"]}
                         )
          )

truep = {"theta.dr00": 0, "theta.dr01": -0.5, "theta.dr10": 0, "theta.dr11": -0.5, "theta.di00": -1, "theta.di01": 0,
         "theta.di10": -1, "theta.di11": 1}
x0 = [1, 2]
sde.simulate(truep=truep, x0=x0)

sde.plot()

qmle = Qmle(sde)

qmle.fit(truep)


# drift function
def b1(x, param):
    out = param["theta.dr0"] - param["theta.dr1"] * x[0]
    return out


def b2(x, param):
    out = param["theta.dr0"] - param["theta.dr1"] * x[1]
    return out


# drift Jacobian
def Jb1_1(x, param):
    out = 1
    return out


def Jb1_2(x, param):
    out = - x[0]
    return out


def Jb2_1(x, param):
    out = 1
    return out


def Jb2_2(x, param):
    out = - x[1]
    return out


# hessian matrices of drift
# block 1
def Hb1_11(x, param):
    out = 0
    return out


def Hb1_12(x, param):
    out = 0
    return out


def Hb1_21(x, param):
    out = 0
    return out


def Hb1_22(x, param):
    out = 0
    return out


# block 2
def Hb2_11(x, param):
    out = 0
    return out


def Hb2_12(x, param):
    out = 0
    return out


def Hb2_21(x, param):
    out = 0
    return out


def Hb2_22(x, param):
    out = 0
    return out


# diffusion functions
def A11(x, param):
    out = param["theta.di0"] + param["theta.di1"] * x[0]
    return out


def A12(x, param):
    out = 0
    return out


def A21(x, param):
    out = 0
    return out


def A22(x, param):
    out = param["theta.di0"] + param["theta.di1"] * x[1]
    return out


# partial derivatives of A
# block 1
def A11_1(x, param):
    out = 1
    return out


def A12_1(x, param):
    out = 0
    return out


def A21_1(x, param):
    out = 0
    return out


def A22_1(x, param):
    out = 1
    return out


# block 2
def A11_2(x, param):
    out = x[0]
    return out


def A12_2(x, param):
    out = 0
    return out


def A21_2(x, param):
    out = 0
    return out


def A22_2(x, param):
    out = x[1]
    return out


# second derivatives of A
# block 11
def A11_11(x, param):
    out = 0
    return out


def A12_11(x, param):
    out = 0
    return out


def A21_11(x, param):
    out = 0
    return out


def A22_11(x, param):
    out = 0
    return out


# block 12
def A11_12(x, param):
    out = 0
    return out


def A12_12(x, param):
    out = 0
    return out


def A21_12(x, param):
    out = 0
    return out


def A22_12(x, param):
    out = 0
    return out


# block 21
def A11_21(x, param):
    out = 0
    return out


def A12_21(x, param):
    out = 0
    return out


def A21_21(x, param):
    out = 0
    return out


def A22_21(x, param):
    out = 0
    return out


# block 22
def A11_22(x, param):
    out = 0
    return out


def A12_22(x, param):
    out = 0
    return out


def A21_22(x, param):
    out = 0
    return out


def A22_22(x, param):
    out = 0
    return out


b_nd = np.array([b1, b2])
A_nd = np.array([[A11, A12], [A21, A22]])

Jb_nd = np.array([[Jb1_1, Jb1_2], [Jb2_1, Jb2_2]])
Hb_nd = np.array([
    [[Hb1_11, Hb1_12], [Hb1_21, Hb1_22]],
    [[Hb2_11, Hb2_12], [Hb2_21, Hb2_22]]
])

DA_nd = np.array([[[A11_1, A12_1], [A21_1, A22_1]],
                  [[A11_2, A12_2], [A21_2, A22_2]]])

HA_nd = np.array([
    [[[A11_11, A12_11], [A21_11, A22_11]],
     [[A11_12, A12_12], [A21_12, A22_12]]],
    [[[A11_21, A12_21], [A21_21, A22_21]],
     [[A11_22, A12_22], [A21_22, A22_22]]]
])

truep = {"theta.dr0": 0, "theta.dr1": -0.5, "theta.di0": -1, "theta.di1": 1}
x0 = np.array([1, 2])


def b(x, param):
    arr_eval = np.empty_like(b_nd)
    it = np.nditer(arr_eval, flags=['multi_index', 'refs_ok'])
    for id in it:
        arr_eval[it.multi_index] = b_nd[it.multi_index](x, param)
    return arr_eval


b(x0, truep)


def A(x, param):
    arr_eval = np.empty_like(A_nd)
    it = np.nditer(arr_eval, flags=['multi_index', 'refs_ok'])
    for id in it:
        arr_eval[it.multi_index] = A_nd[it.multi_index](x, param)
    return arr_eval


def S(x, param):
    out = A(x, param) @ A(x, param).transpose()
    return out


S(x0 + 1, truep)


def Jb(x, param):
    arr_eval = np.empty_like(Jb_nd)
    it = np.nditer(arr_eval, flags=['multi_index', 'refs_ok'])
    for id in it:
        arr_eval[it.multi_index] = Jb_nd[it.multi_index](x, param)
    return arr_eval


Jb(x0, truep)


def Hb(x, param):
    arr_eval = np.empty_like(Hb_nd)
    it = np.nditer(arr_eval, flags=['multi_index', 'refs_ok'])
    for id in it:
        arr_eval[it.multi_index] = Hb_nd[it.multi_index](x, param)
    return arr_eval


Hb(x0, truep)


def DA(x, param):
    arr_eval = np.empty_like(DA_nd)
    it = np.nditer(arr_eval, flags=['multi_index', 'refs_ok'])
    for id in it:
        arr_eval[it.multi_index] = DA_nd[it.multi_index](x, param)
    return arr_eval


DA(x0, truep)


def HA(x, param):
    arr_eval = np.empty_like(HA_nd)
    it = np.nditer(arr_eval, flags=['multi_index', 'refs_ok'])
    for id in it:
        arr_eval[it.multi_index] = HA_nd[it.multi_index](x, param)
    return arr_eval


HA(x0, truep)


def DS(x, param):
    C = np.matmul(DA(x, param), A(x, param).T)
    return C + C.transpose((0, 2, 1))


DS(x0, truep)


def HS(x, param):
    D = np.matmul(HA(x, param), A(x, param).T)
    E = np.swapaxes(np.dot(DA(x, param), DA(x, param).T), 1, 2)
    return D + D.transpose((0, 1, 3, 2)) + E + E.transpose(((0, 1, 3, 2)))


HS(x0, truep)

D0 = np.matmul(HA(x0, truep) + 1, A(x0, truep).T)
D0.transpose((0, 1, 3, 2))
np.swapaxes(D0, 2, 3)
DA0 = DA(x0, truep)
np.swapaxes(np.dot(DA0, DA0), 1, 2)

E = np.swapaxes(np.dot(DA(x, param), DA(x, param).T), 1, 2)

X = np.round(np.random.randn(10).reshape(5, 2) * 10)
X
Xr = X.reshape(5, 1, 2)
DX = Xr[1:len(Xr)] - Xr[:len(Xr) - 1]
dn = 0.1
npar_dr = 2
npar_di = 2
Ss = np.empty((X.shape[0] - 1, X.shape[1], X.shape[1]))
bs = np.empty((X.shape[0] - 1, 1, X.shape[1]))
Jbs = np.empty((X.shape[0] - 1, X.shape[1], npar_dr))
DSs = np.empty((X.shape[0] - 1, npar_dr, X.shape[1], X.shape[1]))

Hbs = np.empty((X.shape[0] - 1, X.shape[1], npar_dr, npar_dr))
HSs = np.empty((X.shape[0] - 1, npar_di, npar_di, X.shape[1], X.shape[1]))

for i in range(len(X) - 1):
    Ss[i] = S(X[i], truep)
    bs[i] = b(X[i], truep)
    Jbs[i] = Jb(X[i], truep)
    DSs[i] = DS(X[i], truep)
    Hbs[i] = Hb(X[i], truep)
    HSs[i] = HS(X[i], truep)

Ss_inv = np.linalg.inv(Ss)
DXS_inv = np.matmul(DX - dn * bs, Ss_inv)

# convention: n: obs; d,e,f,g: dimension (n_var); p,q: params
grad_alpha = 2 * np.matmul(DXS_inv, Jbs)[:, 0, :]
GB1 = np.einsum('nde, npef -> npdf', Ss_inv, DSs)
grad_beta1 = np.trace(GB1, axis1=2, axis2=3)
GB2a = np.einsum('npde, nef -> npdf', GB1, Ss_inv)
grad_beta2 = -1 / dn * np.einsum('npd, nd -> np', np.einsum('nd, npde -> npe', DX[:, 0, :], GB2a), DX[:, 0, :])
grad_beta = grad_beta1 + grad_beta2

A0 = A(X[0], truep)
DS0 = DS(X[0], truep)
Z0 = np.dot(Ss_inv, DSs)
Z0 = np.einsum('bij, bljk -> blik', Ss_inv, DSs)
Z1 = np.trace(Z0, axis1=2, axis2=3)

Z2 = np.einsum('blij, bjk -> blik', np.einsum('bij, bljk -> blik', Ss_inv, DSs), Ss_inv)
Z3 = np.einsum('bi, blik -> blk', DX[:, 0, :], GB2a)

# remember to double check HESSA1 with realistic numbers!!!!
HESSA1 = np.einsum('nd, ndpq -> npq', DXS_inv[:, 0, :], Hbs)
HESSA2 = np.einsum('npd, ndq -> npq', Jbs.transpose((0, 2, 1)), np.einsum('nde, nep -> ndp', Ss_inv, Jbs))
hess_alpha = 2 * HESSA1 - 2 * HESSA2

HESSB1 = np.trace(np.einsum('npde, nqef -> npqdf', GB1, GB1), 3, 4)
HESSB2 = np.trace(np.einsum('nde, npqef -> npqdf', Ss_inv, HSs), 3, 4)

HESSB3a = np.einsum('npde, nef, nqfg -> npqdg', DSs, Ss_inv, DSs)
HESSB3b = np.einsum('nde, npqef, nfg -> npqdg', Ss_inv, HSs, Ss_inv)
HESSB3c = HESSB3a - HESSB3b + HESSB3a.transpose((0, 2, 1, 3, 4))

HESSB3 = np.einsum('nd, npqde, ne -> npq', DXS_inv[:, 0, :], HESSB3c, DXS_inv[:, 0, :])

hess_beta = - HESSB1 + HESSB2 - 1 / dn * HESSB3
# make sure result is symmetric
hess_beta = 0.5 * (hess_beta + hess_beta.transpose((0, 2, 1)))
hess_ab = np.einsum('nd, npde, neq -> npq', DX[:, 0, :], GB2a, Jbs)

hess_i = np.block([[hess_alpha, hess_ab], [hess_ab.transpose((0, 2, 1)), hess_beta]])


def grad_ql(X, param, dn, npar_dr, npar_di):
    # convention: n: obs; d,e,f,g: dimension (n_var); p,q: params
    grad_alpha = -2 * np.matmul(DXS_inv, Jbs)[:, 0, :]
    GB1 = np.einsum('nde, npef -> npdf', Ss_inv, DSs)
    grad_beta1 = np.trace(GB1, axis1=2, axis2=3)
    GB2a = np.einsum('npde, nef -> npdf', GB1, Ss_inv)
    grad_beta2 = -1 / dn * np.einsum('npd, nd -> np', np.einsum('nd, npde -> npe', (DX - dn * bs)[:, 0, :], GB2a),
                                     (DX - dn * bs)[:, 0, :])
    grad_beta = grad_beta1 + grad_beta2

    return 0.5 * np.concatenate([np.sum(grad_alpha, axis=0), np.sum(grad_beta, axis=0)])


def hess_ql(X, param, dn, npar_dr, npar_di):
    # remember to double check HESSA1 with realistic numbers!!!!
    HESSA1 = np.einsum('nd, ndpq -> npq', DXS_inv[:, 0, :], Hbs)
    HESSA2 = np.einsum('npd, ndq -> npq', Jbs.transpose((0, 2, 1)), np.einsum('nde, nep -> ndp', Ss_inv, Jbs))
    hess_alpha = 2 * HESSA1 - 2 * HESSA2

    GB1 = np.einsum('nde, npef -> npdf', Ss_inv, DSs)
    GB2a = np.einsum('npde, nef -> npdf', GB1, Ss_inv)

    HESSB1 = np.trace(np.einsum('npde, nqef -> npqdf', GB1, GB1), axis1=3, axis2=4)
    HESSB2 = np.trace(np.einsum('nde, npqef -> npqdf', Ss_inv, HSs), axis1=3, axis2=4)

    HESSB3a = np.einsum('npde, nef, nqfg -> npqdg', DSs, Ss_inv, DSs)
    HESSB3b = np.einsum('nde, npqef, nfg -> npqdg', Ss_inv, HSs, Ss_inv)
    HESSB3c = HESSB3a - HESSB3b + HESSB3a.transpose((0, 2, 1, 3, 4))

    HESSB3 = np.einsum('nd, npqde, ne -> npq', DXS_inv[:, 0, :], HESSB3c, DXS_inv[:, 0, :])

    hess_beta = - HESSB1 + HESSB2 - 1 / dn * HESSB3
    # make sure result is symmetric
    hess_beta = 0.5 * (hess_beta + hess_beta.transpose((0, 2, 1)))
    hess_ab = np.einsum('nd, npde, neq -> npq', (DX - dn * bs)[:, 0, :], GB2a, Jbs)

    hess_i = -2 * np.block([[hess_alpha, hess_ab], [hess_ab.transpose((0, 2, 1)), hess_beta]])
    return 0.5 * np.sum(hess_i, axis=0)


param = truep
param = qmle.est
grad_ql(X, truep, 0.1, 2, 2)
hess_ql(X, truep, 0.1, 2, 2)

