
# SDElearn: a Python package for SDE modeling

This package implements functionalities for working with Stochastic Differential Equations models (SDEs for short).
It includes simulation routines as well as estimation methods based on observed time series. 
This package is inspired by the R library [yuima](https://yuimaproject.com/).

### Installation
The `sdelearn` package is available on the TestPyPi repository
and can be installed by running

     pip install -i https://test.pypi.org/pypi/ --extra-index-url https://pypi.org/simple sdelearn

## How to create a `sdelearn` class?
The `sdelearn` class is the main class containing the information about a SDE.
Conceptually the information required to describe SDEs can be divided in three groups: _model_, _sampling_ and _data_.
A `sdelearn` class is thus based on three dedicated subclasses, `SdeModel`, `SdeSampling` and `SdeData`, 
containing information about the model, the sampling structure and the observed data respectively. First these three classes 
must be created:

* `SdeModel`: contains information about the Sde model, in particular the "actual" Sde formula. It is assumed to be a parametric 
  model, i.e. the functional form of the model is known up to some parameters. 
In order to construct this class user is required to 
supply two functions, a drift function (`drift`) and a diffusion function (`diff`); an array-like object  `mod_shape` containing the 
dimensions of the model of the form [`n_var`, `n_noise`], where the first dimension represents the number of variables 
and the second the number of Gaussian noises; a dictionary `par_names` with keys `"drift"` and `"diffusion"` and with values 
given by character lists containing all the parameter names appearing in the corresponding drift and diffusion function, 
e.g. `par_names = {"drift": ["par_dr1, "par_dr2"...], "diffusion: ["par_di1, "par_dr2"...] "`(this argument is optional 
and parameter names can be set later using the function `set_param`); a character list `state_var` containing variable names.
Variable names must be supplied in the same order as they appear in the SDE system. If missing, they are assumed to be of the form `x0`, `x1` ... `x[n_var]`. 

  The `mode` argument controls the way the model is specified. There are two ways to supply the drift and diffusion components of the model: "symbolic" or "functional" mode. 

  **_Symbolic mode_**. In symbolic mode (`mode = 'sym'`, the default) the drift and diffusion are supplied as lists of `sympy` expressions, 
where all the non-constant values, i.e. parameters and state variables, are expressed as `sympy` symbols. All the mathematical
functions used in the expressions have to be imported from `sympy`, e.g. use `sympy.sqrt` instead of `math.sqrt`. The length of the `drift`
list has to match number of variables in the model `n_var`. Similarly the `diff` argument has to be a matrix-like object or nested
list with length `n_var` and the length of `diff[0]` is `n_noise`. In this case the dimensions of the model and the parameters are inferred from the expressions
and it is not necessary to specify the `par_names` and `mod_shape` arguments. The variable names are required or assumed to be `x0`, `x1` ... `x[n_var]`.

  **_Function mode_**.
  This is be specified by `mode='fun'`. The **drift function** must be a vector valued function, taking as input two arguments: the state value and the parameters. 
  The input state should be a numeric vector or list, 
  the parameters should be a dictionary. The value returned by this function must match the number of variables `n_var` in the model. 
  Similarly, the **diffusion function** of the model must be supplied as a matrix valued function,
  which takes as input the current state and a dictionary containing the parameters. The dimensions of the output value of the diffusion
  function must match the number of variables and noises supplied: i.e. it must be a `n_var`x`n_noise` matrix.
  Drift and diffusion functions can be scalar valued. 
  The parameters must be addressed by name 
  in both these functions, i.e. as keys in a dictionary. 
  Note that names are important here: names used in the drift and diffusion function definitions must be consistent with
  those supplied as initial values for estimation or simulation (`simulate`). See the examples for details. As a rule of thumb
  the models should be supplied as you'd write them with "pen and paper".
  

  The `options` dictionary allows specifying additional options for compiling the model. It supports the option `hess`, a boolean 
  flag which allows specifying whether to compute second derivatives of the drift and diffusion function while compiling the model.
This is required for exact Hessian computation in `Qmle` (see `Qmle.fit`).

* `SdeSampling`: it contains information about the temporal sampling of the data. It is constructed by supplying the
time of the initial observation `initial` (typically `initial=0`), the last observed time `terminal` and the one between `delta`, the time span between each pair 
  of observations (assumed constant), or `n` the number of points in the grid (including endpoints). If `delta` is given 
the terminal value might not be matched exactly and will be replaced by the largest value in the grid <= terminal. A time grid corresponding to the observation time is automatically generated;
  
* `SdeData`: it contains empirically observed or simulated data. It should be a data frame where each row corresponds to an observation of the time series.
The observation times should match the time grid supplied in the sampling information: that is the number of rows in `SdeData.data`
  should be equal to the length of the grid `Sdesampling.grid`.
  
Finally, an instance of `sdelearn` can be created as `Sde(model = SdeModel, sampling=SdeSampling, data=SdeData)`
where the value of each of the three arguments is an instance of the previous classes. The data argument
is optional. Data can be added later e.g. by simulation or by using the `setData` method. 


## Learning model parameters using a `SdeLearner`

The parameters of a SDE can be estimated using an object of class
`SdeLearner`. 
A learner object is built around a Sde. Some learners (e.g. AdaLasso) 
require information about an initial estimate. This is provided by
supplying a fitted learner as `base_estimator` when the object is created.
Details on the parameters depend on the specific implementation of each learner.
Every Learner implements the following methods:

* `loss`: computes the loss function associated with the learner (see specific implementations for details)
* `fit`: learns the model parameters by minimizing the loss function
* `predict`: estimates the trend of the series on a given set of times
* `gradient`, `hessian`: compute the exact gradient and hessian of 
the loss function. Available only in symbolic mode.

Every Learner has the following fields:
* `est`: dictionary of estimated parameters
* `vcov`: (asymptotic) covariance matrix, based on observed Fisher information
* `sde`: base Sde object
* `optim_info`: dictionary containing info about the optimization process (e.g. number of iteration, method used, etc.)



_Currently available learners are Qmle for a quasi-likelihood-based estimator, AdaLasso, AdaBridge, AdaEnet classes for regularized estimation._

### Qmle
SdeLearner based on quasi maximum-likelihood estimation.
```
qmle = Qmle(sde)
```
Parameters: 
* `sde`: object of class Sde

#### Details
The loss function is the negative log-quasi-likelihood.
If the Sde object was specified in `symbolic` mode, 
symbolic derivatives of the loss functions are computed.

The `fit` method allows for minimization of the "full" loss
function or "two-step", estimating diffusion parameters first and then 
drift. The `hess_exact` option controls how the Hessian matrix (used to build the Fisher information)
is computed. It allows to choose 
whether to use the exact symbolic computation of the Hessian matrix 
or a much faster computation based on the outer product of the gradient at the minimum point. 
(Both converge to the true information matrix when rescaled, see De Gregorio and Iacus).
Note that exact Hessian computation is available only if the model has been built with the second derivatives' computation.
_The recommended setting is to use `two_step=True` and `hess_exact=False` for numerical stability_

If a two-step (adaptive) estimation procedure is 
used, then the `loss2` and `gradient2` functions can be used to get the loss and the gradient, respectively, for drift and diffusion parameters only.
The loss functions are as in Uchida, Yoshida (2012). 
The parameter group can be specified via the `group` parameter, which can take values `alpha` for drift or `beta` for diffusion parameters. 



The loss and the gradient are by default scaled by the number of observation. The gradient can optionally be scaled by its asymptotic rates
(see e.g. Kessler 1997) by specifying `asy_scale=True`. This is used in the OPG approximation of the Hessian, in order to obtain entries 
that have a consisten scale.

See `sde.model` options for details. 

### AdaLasso, AdaBridge, AdaElasticNet
SdeLearner based on Least Square Approximation (LSA) of the loss function.
It requires a base estimator of class SdeLearner(e.g. Qmle) to get 
an initial estimate and the Fisher information and to build 
the penalized LSA objective function. 
AIC computation is used for choosing the best tuning parameter
or, cv based method of a KL based method. 




## Technical details

This section contains some information about the internal structure of the package
(if you are getting unexpected errors this is a good place to start).

* `param`: when in `mode='fun'`, typical name for parameter argument of drift and diffusion function. Both functions share the same 
parameter dictionary, and the full parameter dictionary will be passed to both functions. Parameter names
used inside the function will make the difference. Initially, if the `par_names` argument is left blank, the `model` is not aware of what the parameters 
of the models are. They will be inferred when simulation takes place without distinction between drift and diffusion parameters.
When the `simulate` method 
or an estimation method is called the user will have to supply a `truep` parameter or a starting parameter for 
the optimization which will act as a _template_ for the parameter space of the model.
Before any estimation takes place the parameter names should be explicitly set.


* The `SdeLearner` class is generic ("abstract") and the user should never
directly use it, but instead they should use one of the subclasses implementing 
specific methods.

* In numerical computations, it is important that the dictionary of parameters is ordered.
Fit and loss functions should automatically match the supplied values
with the order specified in the model: currently automatic reordering is done for
arguments `param` of the loss function, `start` and `bounds` in model fitting, `param` in simulate. Note that bounds
do not have names, so they are assumed to have the same order as `start`.
The ordered list of parameters can be accessed by `Sde.model.param`.

* In the `gradient2(param, group='beta')` case, a full parameter vector has to be supplied, even if the drift components 
 actually are not used in the computation.

* in symbolic mode, the model computes parameter and variable maps, to keep track of which parameters and variables
appear in each equation, for both drift and diffusion components. These are called `par_map_drift`, `var_map_diff`, etc.

* In the case of the `two_step` quasi-likelihood estimation, if the estimation problem for the diffusion can be diagonalized 
(i.e. the diffusion matrix is diagonal, each parameter appears only in one equation, and depends only on one variable), 
a significant speedup is achieved by fitting one equation at a time. This is automatically checked by the `check_diag_est` 
method when `fit` is called. This uses the maps defined above. 

* Qmle has a low-memory mode that splits the gradient computation in batches. 
This avoids issues memory limit being exceeded for larger models, resulting in 
the process being killed. To turn this on, use the method `set_low_mem(switch=True)`. 
The instance of Qmle enters in low memory mode and all the subsequent computations will be split.
Note that this still returns the exact gradient, i.e. all the batches will be subsequently processed. 
This saves memory, at the cost of a longer execution time.

## Examples

Fit a multivariate SDE model from simulated data (script [sdelearn-test.py](tests/sdelearn-test.py)).

**Functional mode**. This is the direct way to approach Sde modeling with `sdelearn`.
Import the `sdelearn` libray

    from sdelearn import *
    
Define the drift function:

    def b(x, param):
        out = [0,0]
        out[0]= param["theta_dr00"] - param["theta_dr01"] * x[0]
        out[1] = param["theta_dr10"] - param["theta_dr11"] * x[1]
        return out



Define the diffusion function:

    def A(x, param):
        out = [[0,0],[0,0]]
        out[0][0] = param["theta_di00"] + param["theta_di01"] * x[0]
        out[1][1] = param["theta_di10"] + param["theta_di11"] * x[1]
        out[1][0] = 0
        out[0][1] = 0
        return out


Create the Sde object, specifying the parameters used in the drift and diffusion functions 

    par_names = {"drift": ["theta_dr00", "theta_dr01", "theta_dr10", "theta_dr11"],
                 "diffusion": ["theta_di00", "theta_di01", "theta_di10", "theta_di11"]}

    sde = Sde(sampling=SdeSampling(initial=0, terminal=2, delta=0.01),
              model=SdeModel(b, A, mod_shape=[2, 2], par_names=par_names, mode='fun'))
    
    print(sde)


Set the true value of the parameter and simulate a sample path of the process:

    truep = {"theta_dr00": 0, "theta_dr01": -0.5, "theta_dr10": 0, "theta_dr11": -0.5, "theta_di00": 0, "theta_di01": 1, "theta_di10": 0, "theta_di11": 1}
    sde.simulate(truep=truep, x0=[1, 2])


Plot the simulated path:

    sde.plot()

Fit the model using a quasi-maximum-likelihood estimator:

    qmle = Qmle(sde)

    # generate some random starting values
    all_param = [p for k in par_names.keys() for p in par_names.get(k)]
    n_param = len(all_param)
    startp = dict(zip(all_param, np.round(np.abs(np.random.randn(n_param)), 1)))
    
    qmle.fit(startp, method='BFGS')

See the results: estimated parameter, its variance covariance matrix and information about the optimization process
   
    qmle.est
    qmle.vcov
    qmle.optim_info

Compute and show predictions (estimated trend)

    qmle.predict().plot()


**Symbolic mode**.

    from sdelearn import *
    import numpy as np
    import sympy as sym


Create symbols and define the drift vector and diffusion matrix.

    n_var = 2
    theta_dr = [sym.symbols('theta_dr{0}{1}'.format(i, j)) for i in range(n_var) for j in range(2)]
    theta_di = [sym.symbols('theta_di{0}{1}'.format(i, j)) for i in range(n_var) for j in range(2)]

    all_param = theta_dr + theta_di
    state_var = [sym.symbols('x{0}'.format(i)) for i in range(n_var)]

    b_expr = np.array([theta_dr[2*i] - theta_dr[2*i+1] * state_var[i] for i in range(n_var)])

    A_expr = np.full((n_var,n_var), sym.sympify('0'))
    np.fill_diagonal(A_expr, [theta_di[2*i] + theta_di[2*i+1] * state_var[i] for i in range(n_var)])

Instanciate the Sde. Note that in this case it is not necessary to specify `par_names` 
while `mode='sym` is the default.


    sde = Sde(sampling=SdeSampling(initial=0, terminal=20, delta=0.01),
              model=SdeModel(b_expr, A_expr, state_var=[s.name for s in state_var]))
    print(sde)

Fix some paramter value and simulate:

    truep = dict(zip([s.name for s in all_param], np.round(np.abs(np.random.randn(len(all_param))), 1)))
    sde.simulate(truep=truep, x0=np.arange(n_var))

    sde.plot()

Fit the data using qmle, specifying box-constraints for the optimization


    qmle = Qmle(sde)
    startp = dict(zip([s.name for s in all_param], np.round(np.abs(np.random.randn(len(all_param))), 1)))
    box_width = 10
    bounds = [(-0.5*box_width, 0.5*box_width)]*len(all_param) + np.random.rand(len(all_param)*2).reshape(len(all_param), 2)


    qmle.fit(start=startp, method='L-BFGS-B', bounds = bounds)
    qmle.est
    qmle.optim_info


Fit the parameters using adaptive lasso and qmle as initial estimator.
Set a delta > 0 value to use adaptive weights, additionally use
the weights argument to apply a specific penalty to each parameter.

    lasso = AdaLasso(sde, qmle, delta=1, start=startp)
    lasso.lambda_max
    lasso.penalty
    lasso.fit()

By default no lambda value is chosen and the full path of estimates is computed:
    
    lasso.est_path
    lasso.plot()

In order to choose a penalization value fit using last 10% obs as validation set (optimal lambda minimizes validation loss):
    
    lasso.fit(cv=0.1)

In this case the estimate corresponding to optimal lambda is computed:

    lasso.est
    lasso.vcov



