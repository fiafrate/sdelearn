# SDElearn: a Python package for SDE modeling

SDElearn implements simulation and estimation for parametric stochastic differential equations observed at discrete times. The model, the sampling scheme and the observed trajectory are represented separately, following the approach of the R library [yuima](https://yuimaproject.com/).

## Installation

```sh
git clone https://github.com/fiafrate/sdelearn.git
cd sdelearn
python -m pip install .
```

## Model, sampling and data

Let $X$ be a $d$-dimensional process satisfying

$$
dX_t = b(X_t,\alpha)\,dt + A(X_t,\beta)\,dW_t,
$$

where $W$ is an $r$-dimensional Brownian motion. Supply the $d$ drift components and the $d\times r$ diffusion coefficient $A$. We write $\theta=(\alpha,\beta)$ and denote the numbers of drift and diffusion parameters by $p_\alpha$ and $p_\beta$.

We observe $X$ at $n+1$ equidistant times $t_0,\ldots,t_n$, including the starting point. These observations give $n$ increments. An `Sde` combines three objects.

* `SdeModel` holds the coefficients and parameter names. In symbolic mode (`mode="sym"`, the default), supply lists or arrays of SymPy expressions. The model infers dimensions and parameter names. Supply state variable names in equation order, or use the defaults `x0`, `x1`, etc. Use SymPy functions in these expressions. In functional mode (`mode="fun"`), supply two callables `b(x, param)` and `A(x, param)`, the dimensions `mod_shape=(d, r)` and, for estimation, `par_names={"drift": [...], "diffusion": [...]}`. Both callables receive the full parameter dictionary.
* `SdeSampling(initial, terminal, n=...)` takes the number of observation times as its `n` argument. With the notation above, `sampling.n` is $n+1$. Alternatively, supply `delta` for the time step. The last grid point is then the largest admissible time up to `terminal`.
* `SdeData` holds one observation per row and one state variable per column. Supply observed data to `sde.set_data(data)`, or use `sde.simulate(param, x0)` to generate a trajectory by the Euler scheme. The model and sampling scheme can also be specified before the data are available.

## Learning model parameters using a `SdeLearner`

An `Sde` describes the stochastic differential equation, the observation times and the trajectory. A learner specifies how to estimate the parameters from those observations. Several learners can therefore be fitted to the same `Sde`. Each keeps its own estimates, so their fits and predictions can be compared on the same trajectory.

All estimators provided by SDElearn derive from `SdeLearner` and share the following methods and fields.

**Learner methods**

- `loss` evaluates the estimation criterion at the supplied parameter values.
- `fit` estimates the model parameters and stores the results on the learner.
- `predict` simulates trajectories under the fitted parameters to estimate the mean trajectory on a chosen time grid.
- `gradient` and `hessian` evaluate derivatives when the estimation procedure provides them. Symbolic `Qmle`, for example, provides both for the quasi-likelihood loss.

**Learner fields**

- `sde` holds the model, sampling scheme and trajectory used for estimation.
- `est` is the dictionary of fitted coefficients, keyed by parameter name.
- `vcov` holds the covariance matrix when available. Regularized estimators inherit it from the base learner.
- `optim_info` records information about the fit, such as the optimizer result and the fitted Hessian when available.

The regularized estimators `AdaLasso`, `AdaElasticNet` and `AdaBridge` each require an explicitly chosen base learner. For example, passing a fitted `Qmle` to `AdaLasso` supplies the initial estimate and Hessian for a quadratic approximation to the quasi-likelihood loss. With direct penalization, the base learner supplies the loss itself. An unfitted base learner is fitted automatically when the chosen procedure needs an initial estimate.

### Example: simulation and estimation

Consider a mean-reverting diffusion with drift $a-bx$ and unit diffusion coefficient. The following example is used throughout the guide.

```python
import numpy as np
import sympy as sym
from sdelearn import AdaBridge, AdaElasticNet, AdaLasso, Qmle, Sde, SdeModel, SdeSampling

np.random.seed(42)
x, a, b = sym.symbols("x a b")
model = SdeModel([a - b*x], [[1]], state_var=["x"], options={"hess": True})
sampling = SdeSampling(initial=0, terminal=30, delta=0.05)
sde = Sde(sampling=sampling, model=model)
sde.simulate(param={"a": 0.4, "b": 1.2}, x0=[0.0])

start = {"a": 0.1, "b": 1.0}
qmle = Qmle(sde).fit(start=start, hess_exact=True)
estimate = qmle.est
covariance = qmle.vcov
future = SdeSampling(initial=30, terminal=31, delta=0.05)
mean_path = qmle.predict(sampling=future, x0=sde.data.data.iloc[-1].to_numpy(), n_rep=100)
```

The same observations can be fitted in functional mode. A drift returns a vector of length $d$ and a diffusion returns a matrix of shape $(d,r)$, including in a scalar model.

```python
def drift(x, param):
    return [param["a"] - param["b"] * x[0]]

def diffusion(x, param):
    return [[1.0]]

functional = SdeModel(drift, diffusion, mode="fun", mod_shape=(1, 1),
                      par_names={"drift": ["a", "b"], "diffusion": []})
sde_fun = Sde(sampling=sampling, model=functional).set_data(sde.data.data.to_numpy())
functional_loss = Qmle(sde_fun).loss(estimate)
```

`model.param` gives the order used in numerical arrays and matrices. Drift parameters precede diffusion parameters, with names sorted within each group. QMLE reorders the starting dictionary to model order. Supply any parameter bounds in the order of the starting dictionary. For regularized estimation, supply parameter vectors, weight dictionaries and solver bounds in model order.

## Quasi-likelihood estimation

Let $S(x,\beta)=A(x,\beta)A(x,\beta)^\top$ denote the instantaneous covariance matrix. Write $h=t_{i+1}-t_i$, $\Delta_i X=X_{t_{i+1}}-X_{t_i}$, $b_i=b(X_{t_i},\alpha)$, $S_i=S(X_{t_i},\beta)$ and $r_i=\Delta_i X-hb_i$. The conditional Gaussian approximation has mean increment $hb_i$ and covariance $hS_i$. Up to parameter-independent constants, `Qmle.loss(param)` returns the mean negative quasi-log-likelihood

$$
L(\theta)=\frac{1}{2n}\sum_{i=0}^{n-1}
\left(\log\det S_i+\frac{r_i^\top S_i^{-1}r_i}{h}\right).
$$

The same full-data normalization applies in functional and symbolic mode. In functional mode the coefficients are evaluated observation by observation and linear systems give the quadratic forms. Estimation uses the joint objective and a SciPy minimizer (default `method="BFGS"`). The fit uses the optimizer's inverse-curvature output, so the method must supply `hess_inv`. Bounds can be supplied to a method supporting them, for instance `"L-BFGS-B"`.

### Joint and two-step estimation

In symbolic mode, `two_step=False` with both parameter groups present minimizes $L$ jointly. The default `two_step=True` first estimates $\beta$ from the increments and then estimates $\alpha$ with the diffusion fixed.[^two-step] The two objectives are

$$
\begin{aligned}
L_\beta(\beta)
  &= \frac{1}{2n}\sum_i \left(\log\det S_i
     + \frac{(\Delta_i X)^\top S_i^{-1}\,\Delta_i X}{h}\right),\\
L_\alpha(\alpha;\hat{\beta})
  &= \frac{1}{2nh}\sum_i r_i^\top S(X_{t_i},\hat{\beta})^{-1}r_i.
\end{aligned}
$$

With a single parameter group the SciPy fit takes the corresponding step. Pass the full parameter dictionary to `loss2(param, group)` and `gradient2(param, group)`. Use `"alpha"` for drift and `"beta"` for diffusion. In a model with drift, the second derivative of $L_\beta$ can differ from the diffusion block of the joint Hessian because the former uses raw increments.

**Diagonal diffusion.** In a multivariate two-step fit, `check_diag_est()` detects a square diagonal $A$ for which each $A_{ii}$ depends only on $x_i$ or is state-independent, and each diffusion parameter belongs to a single equation. Then $S_{ii}=A_{ii}^2$ and $L_\beta$ separates into scalar objectives. The fitter estimates each parameterized row on its own trajectory and collects all its coefficients by name. For the approximate information matrix it places each scalar-fit block in the corresponding rows and columns of the parent diffusion matrix. This saves evaluating the full multivariate diffusion-derivative tensors during those scalar fits. The drift step can still include interactions among all state variables. A general factor with diagonal $AA^\top$ goes through the joint diffusion calculation.

For a bivariate example, consider two mean-reverting equations with a common sampling scheme and distinct diffusion parameters. Here each diagonal loading stays positive over the state space. The state dependence in the first row allows two diffusion coefficients to be estimated from that scalar equation.

```python
x1, x2, k1, k2, s1, s2, c1 = sym.symbols("x1 x2 k1 k2 s1 s2 c1")
model2 = SdeModel([-k1*x1, -k2*x2], [[s1 + c1*x1**2/(1 + x1**2), 0], [0, s2]],
                  state_var=["x1", "x2"])
sde2 = Sde(sampling=SdeSampling(0, 30, delta=0.05), model=model2)
truth = {"k1": 1.0, "k2": 0.8, "s1": 0.7, "s2": 1.0, "c1": 0.3}
sde2.simulate(param=truth, x0=[1.0, 0.0])
fit2 = Qmle(sde2).fit(start=truth, method="L-BFGS-B",
                       bounds=[(0.05, 3.0)] * len(truth))
diffusion_estimates = {name: fit2.est[name] for name in model2.diff_par}
```

### Scores, Hessians and covariance

The gradient and Hessian use the same normalization as the loss (division by $n$) for numerical stability. Let $g_i$ denote the gradient of the negative quasi-log-likelihood contribution of increment $i$. With `ret_sample=True`, row $i$ is $g_i^\top/n$. For statistical comparisons, `asy_scale=True` divides the unnormalized drift and diffusion gradients by their respective asymptotic rates, $\sqrt{nh}$ and $\sqrt{n}$.

With $J_i=\partial_\alpha b_i$ and $D_{ij}=\partial_{\beta_j}S_i$, the joint score contributions are

$$
\begin{aligned}
g_{i,\alpha} &= -J_i^\top S_i^{-1}r_i,\\
g_{i,\beta_j} &= \frac{1}{2}\left[
\mathrm{tr}(S_i^{-1}D_{ij}) - \frac{r_i^\top S_i^{-1}D_{ij}S_i^{-1}r_i}{h}\right].
\end{aligned}
$$

For the first diffusion step, replace $r_i$ by the raw increment $\Delta_i X$. Caching $S_i^{-1}$ and $r_i^\top S_i^{-1}$ avoids repeating these calculations when evaluating the loss and gradient at the same parameters.

By default (`hess_exact=False`), the fit uses the outer-product approximation $H = n^{-1}\sum_{i=1}^n g_i g_i^\top$. The two-step fit forms the drift and diffusion blocks separately using the respective group scores.

To obtain the exact Hessian, construct the symbolic model with `options={"hess": True}` and request `fit(..., hess_exact=True)`. The model otherwise skips symbolic second derivatives and the fit uses the score approximation. `hessian(param, group="all")` returns the Hessian of the mean joint loss including the mixed block. The `"alpha"` and `"beta"` groups differentiate the corresponding two-step objectives. A two-step fit assembles the exact group blocks, while a simultaneous fit uses the joint Hessian. Both options store $H$ in `optim_info["hess"]` and return $H^{-1}/n$ in `vcov` when the inverse exists.

```python
H = qmle.hessian(estimate)
H_drift = qmle.hessian(estimate, group="alpha")
scores = qmle.gradient(estimate, ret_sample=True)
batch_scores = qmle.gradient(estimate, batch_id=[4, 1], ret_sample=True)
```

**Symbolic and numerical calculations.** `SdeModel` differentiates the coefficient expressions once and compiles numerical functions with SymPy. At the observed states NumPy evaluates $b$, $A$, the drift Jacobian `Jb` and the diffusion derivative `DA`. With second derivatives enabled it also evaluates `Hb` and `HA` as required. For instance, for a diffusion parameter $\beta_j$ the calculation uses

$$
\partial_j S_i=(\partial_j A_i)A_i^\top+A_i(\partial_j A_i)^\top .
$$

Array contractions sum over state and noise coordinates while retaining the increment and parameter indices. For a batch of $B$ increments, the diffusion values have shape $(B,d,r)$, `DA` has shape $(B,p_\beta,d,r)$ and `HA` has shape $(B,p_\beta,p_\beta,d,r)$. Group-specific Hessians evaluate derivatives for the requested group.

The symbolic learner stores observation arrays in single precision. Choose a parameter domain on which $S_i$ is positive definite along the observed path. Singular covariance evaluations and nonpositive determinants can produce large objective values during minimization. The QMLE object retains the observations `X` and increments `DX`. `update_aux` caches `As`, `Ss`, `Ss_inv`, `bs` and the weighted residuals `DXS_inv` at the current parameters and batch. `update_aux2` updates the quantities needed for each step, reusing the diffusion covariance when only the drift changes. Repeated evaluations with compatible group, parameters and batch reuse these arrays. Evaluated derivative tensors such as `Jb` and `DA` are local to the gradient or Hessian call. `set_low_mem(True)` evaluates gradient derivatives in successive groups of 1,000 increments, retaining the full contribution of the requested sample. For an explicit `batch_id`, returned score rows follow the supplied increment order.

**Experimental accelerated gradient descent.** In symbolic mode `fit(start, method="AGD")` calls `SdeLearner.nesterov_descent` on the joint objective. It records the returned estimate and computes its full-data curvature and covariance under the same options as a simultaneous symbolic SciPy fit. Full-vector, cyclic and sampled updates are available in the experimental routine. With `batch` set to a fraction, the gradient and the subsequent line search can use different draws, and the stopping rule uses the sampled gradient. For the estimation procedures and examples in this guide the default optimizer is SciPy BFGS.

## Regularized estimation

`AdaLasso`, `AdaElasticNet` and `AdaBridge` estimate a regularization path for an explicitly chosen `base_estimator`. The base learner defines the loss $L_0$ and, when required, an initial estimator $\tilde{\theta}$ and its curvature matrix $H$. A fitted learner supplies `est` and `optim_info["hess"]`. An unfitted learner is fitted automatically when an initial estimate is required. Supply its fitting arguments, e.g. `start=start`, to the regularizer constructor. The explicit string `"Qmle"` also constructs a QMLE. Other learners can be supplied as instances.

### Initial estimator and penalized loss

Write the criterion as $F_\lambda(\theta)=f(\theta)+\lambda P(\theta)$. For the least squares approximation the smooth loss is

$$
f(\theta)=\frac{1}{2}(\theta-\tilde{\theta})^\top H(\theta-\tilde{\theta}).
$$

The constructor options determine which smooth loss and weights are used.

| Constructor options | Smooth loss $f$ | Initial estimation |
| --- | --- | --- |
| `lsa=True` (default) | Quadratic approximation | Uses the least squares approximation and enables adaptive initialization. |
| `lsa=False, adaptive=True` | Base loss $L_0(\theta)$ | Uses the initial estimate for weights and its curvature for step sizes. |
| `lsa=False, adaptive=False` | Base loss $L_0(\theta)$ | Uses fixed weights and first derivatives. Leaves `ini_hess` as `None`. |

The constructor symmetrizes the initial curvature. With `hess_check=True` it replaces each strictly negative eigenvalue by $0.001\,|Z|$ for an independent standard normal draw $Z$. A zero eigenvalue can still give a singular matrix. The same option is applied to each training Hessian during validation.

Let $u_j$ be the supplied `weights` (default one). In the adaptive case $w_j=u_j/|\tilde{\theta}_j|^\delta$ with `delta=0` by default. For positive $\delta$, zero initial coefficients give singular weights. Supply the weights in initial-estimate order. In the fixed-weight case $w_j=u_j$ and the entries are matched by parameter name. The three penalties are

$$
\begin{aligned}
P_{\mathrm{Lasso}}(\theta)&=\sum_j w_j|\theta_j|,\\
P_{\mathrm{EN}}(\theta)&=\eta\sum_j w_j|\theta_j|+\frac{1-\eta}{2}\sum_j\theta_j^2,\\
P_{\mathrm{Bridge}}(\theta)&=\sum_j w_j|\theta_j|^q .
\end{aligned}
$$

In `AdaElasticNet` the mixing coefficient $\eta$ is the argument `alpha`, with $0<\eta\le1$ and `q=1`. At $\eta=1$ the penalty is the adaptive Lasso.[^elastic-net] `AdaBridge` implements the half-power threshold, `q=0.5`.[^bridge]

### Choice of the penalty grid

Let $g=\nabla f(0)$. For the least squares approximation $g=-H\tilde{\theta}$. For direct-loss estimation the constructors that calculate a backward endpoint evaluate the base gradient at zero. This requires a loss defined at that parameter, which deserves attention if zero makes the diffusion covariance singular. For positive weights the Lasso and Elastic Net endpoints follow from the coordinatewise optimality condition at zero:

$$
\begin{aligned}
\lambda_{\max}^{\mathrm{Lasso}} &= \max_j \frac{\lvert g_j\rvert}{w_j},\\
\lambda_{\max}^{\mathrm{EN}} &= \max_j \frac{\lvert g_j\rvert}{\eta w_j}.
\end{aligned}
$$

Bridge has distinct forward and backward paths.[^pathwise] Let $L$ be the largest eigenvalue of $H$ and $c_q=[2(1-q)]^{1/(2-q)}[1+q/(2(1-q))]$. For the backward path the threshold at zero gives the reference endpoint

$$
\lambda_{\max}^{\mathrm{BW}}=c_q^{\,q-2} L^{q-1}
\max_j\frac{|g_j|^{2-q}}{w_j}.
$$

This is a zero fixed-point condition for a nonconvex update. The forward path starts from the initial estimate. For the quadratic approximation the endpoint is obtained from the last active coordinate, using $h_j=H_{jj}$ and

$$
\begin{aligned}
a_{qj} &= (q w_j)^{1/(2-q)}\left[
    h_j^{(1-q)/(2-q)}(1-q)^{1/(2-q)}
    + \left(\frac{1-q}{h_j}\right)^{(q-1)/(2-q)}\right],\\
\lambda_{\max}^{\mathrm{FW}} &=
    \left(\max_j \frac{\lvert(H\tilde{\theta})_j\rvert}{a_{qj}}\right)^{2-q}.
\end{aligned}
$$

The automatically generated grid contains zero and `n_pen - 1` logarithmically spaced positive values (default `n_pen=100` for a positive endpoint). The first positive value is $\min(\lambda_{\max}/n_{\mathrm{pen}}^2,10^{-5})$. An explicit `penalty` array is sorted and determines the number of rows. Use at least three distinct grid points for selection. Each solved row supplies the starting value for the next penalty. On adaptive backward paths the endpoints are initialized at the initial estimate and the zero vector, so a custom grid should include zero and a sufficiently large upper endpoint.

Lasso and Elastic Net traverse the grid from largest to smallest. Adaptive Bridge uses the backward path by default and accepts `fit(backwards=False)` for a forward path. Automatic forward grid construction requires the quadratic approximation. With `lsa=False, adaptive=False`, Bridge requires an explicit grid and fits it in ascending order, including the smallest penalty. The path direction is carried through to validation fits.

### Numerical methods

At each penalty value, the solver takes a gradient step for the smooth loss and then applies the thresholding rule for the chosen penalty. The solution initializes the fit at the next point on the grid. For Bridge, the objective is nonconvex, so the forward and backward paths can yield different estimates at the same penalty.[^pathwise]

Use `opt_alg` in `fit(...)` to choose how the coefficients are updated.

| `opt_alg` | Method |
| --- | --- |
| `"fista"` | Default for Lasso and Elastic Net. Extrapolates from the two most recent iterates before updating all coefficients together. |
| `"mAPG"` | Default for Bridge. Takes a step from both the current and the extrapolated iterate and retains the candidate with the smaller penalized loss. |
| `"cyclic"` | Updates one coefficient at a time, cycling through all coordinates. Available for all three estimators. |
| `"block_wise"` | Updates one parameter group at a time, cycling through the groups defined by the model. Available for all three estimators when the model has multiple groups. |

| Argument to `fit(...)` | Effect |
| --- | --- |
| `backtracking` | `True` enables a line search that shortens the step when the smooth loss exceeds its local quadratic bound. `False` disables the search. The default (`None`) uses the initial Hessian to set the step when available and enables backtracking otherwise. |
| `s_ini` | Initial trial step for backtracking (default 10). Also sets the fixed step when backtracking is disabled and no initial Hessian is available. |
| `epsilon` | Stopping tolerance for the change in the coefficient vector, assessed after a full sweep for coordinate and block updates. Defaults to $10^{-6}$ for Lasso and $10^{-3}$ for Bridge and Elastic Net. |
| `max_it` | Maximum number of updates at each penalty value (default 1,000). |
| `bounds` | Optional array with one lower and upper bound per coefficient, in model parameter order. The default is unconstrained. |

### Selecting an estimate from the path

A successful `fit()` always selects a penalty. Let $K$ be the number of grid points, $\hat{\theta}_\lambda$ the full-data estimate and $d_\lambda$ the number of its exactly nonzero coefficients. With `cv=None` the criterion on the $K-2$ interior penalties is

$$
C_k(\lambda)=L_0(\hat{\theta}_\lambda)+ k\,\frac{d_\lambda}{n+1}.
$$

Selection uses the base loss even when the path is constructed by least squares approximation. `aic_k=1` gives the default AIC-style criterion. The multiplier can be adjusted to the scale of a user-supplied loss. For the QMLE normalization $L_0=-\ell/n$, multiplying $C_k$ by $2n$ gives the ordinary AIC penalty $2d_\lambda$ at $k=(n+1)/n$. For BIC with $n$ increments as sample size, use $k=(n+1)\log(n)/(2n)$. Both choices use `fit(aic_k=k)`.

| `cv` | Observations and selection score |
| --- | --- |
| `None` | Full-data information criterion $C_k$. |
| A fraction in $(0,1)$ | Chronological holdout of the final $\lfloor (n+1)\,cv\rfloor$ observations. The final training observation is also the first validation observation. |
| `True` | Expanding-window cross-validation with `nfolds=5` by default. Start with $\lfloor 0.4(n+1)\rfloor$ training observations, then extend the training prefix by successive windows. The last window includes the remainder. Adjacent training and validation sets share their boundary observation. |

Choose windows with enough observations to form increments and estimate the training model. Validation uses the full-data penalty grid. Each training fit retains the chosen loss, weighting options and curvature adjustment. Adaptive fits re-estimate the base learner on the training prefix using the full-data initial estimate as the starting value. The learner also computes the coefficient path on the full sample.

With `cv_metric="loss"` the score is the base loss on held-out observations at the training coefficients. With `cv_metric="mse"` it is the mean squared difference across validation times and components between the observations and a Monte Carlo mean trajectory starting at the boundary observation. `n_rep` controls the number of simulated paths per prediction (default 100). The nonzero count enters only the information criterion. Fold scores are averaged over available values for each penalty. If every penalty is unscorable, selection raises an error.

`lambda_min` is the minimizing interior penalty (first minimizer in ascending order), and `est` is `coef(lambda_min)` from the full-data path. `lambda_opt` is the largest interior penalty whose score is strictly below the minimum plus the following tolerance:

* For the information criterion, the midpoint-quantile median score minus the minimum.
* For a single validation set, one population standard deviation across the interior scores, omitting missing values.
* For expanding windows, a separate tolerance for each penalty, equal to its fold-score standard deviation divided by $\sqrt{\texttt{nfolds}}$. The divisor uses the requested number of folds.

If the strict comparison leaves no eligible penalty, `lambda_opt` is the largest minimizing penalty. Use `coef(lambda_opt)` to retrieve its coefficients.

```python
lasso = AdaLasso(sde, base_estimator=qmle, delta=1, n_pen=20).fit()
n = sampling.n - 1
bic_k = (n + 1) * np.log(n) / (2 * n)
lasso.fit(aic_k=bic_k)
bic_estimate = lasso.est.copy()

enet = AdaElasticNet(sde, qmle, alpha=0.6, n_pen=20).fit(cv=0.2)
bridge = AdaBridge(sde, qmle, q=0.5, n_pen=20).fit(cv=True, nfolds=3)
alternative = bridge.coef(bridge.lambda_opt)

direct = AdaBridge(sde, Qmle(sde), lsa=False, adaptive=False,
                   penalty=[0, 0.002, 0.01, 0.05, 0.2]).fit()
```

## Returned quantities

For $p=p_\alpha+p_\beta$, coefficient dictionaries use model parameter names. Matrix axes follow the same order. A few methods update the object in place, as indicated below.

| Model, sampling and data | Return value |
| --- | --- |
| `SdeModel.set_param`, `set_var_names` | Update parameter groups or state names in place and return `None`. `set_der()` returns the symbolic expressions for the coefficient functions and their derivatives. |
| `SdeSampling.sub_sampling(...)` | New sampling object. Supply one of `first_n`, `last_n`, `from_range=[i,j]` (indices $i$ through $j-1$), `new_init` or `new_term`. The latter two use the nearest grid index as a slice boundary. |
| `SdeData.format_data(...)` | Updates the time index and column names of the data frame in place and returns `None`. |
| `Sde.set_model`, `set_sampling`, `set_data` | Return the `Sde` object with the corresponding component assigned. |
| `Sde.simulate(param, x0)` | Stores the simulated data and returns the `Sde`. With `ret_data=True` returns a numerical array of shape $(n+1,d)$. Optional `bounds` apply to the state during simulation. |
| `Sde.plot(save_fig=None)` | Plots the observed trajectory or saves the figure and returns the `Sde`. |
| `learner.predict(sampling=None, x0=None, n_rep=1000)` | Time-indexed data frame of Monte Carlo mean trajectories under `est`. With `ret_data=True` returns the individual paths of shape $(n_{\mathrm{rep}},n+1,d)$ on the chosen grid. |

| QMLE | Return value |
| --- | --- |
| `fit(start, ...)` | The fitted learner, with `est`, `vcov` and `optim_info`. Joint fits store the optimizer result in `res`. Two-step fits store `res_alpha` and `res_beta`, with per-equation information for diagonal diffusion. `optim_info["hess"]` holds the curvature used for inference. |
| `loss`, `loss2` | Scalar mean negative quasi-likelihood or selected two-step objective. |
| `gradient`, `gradient2` | Vector of length $p$ or the selected group size. `ret_sample=True` returns a matrix with one row per requested increment. |
| `hessian`, `rates` | Square matrix for all parameters or the selected group. An empty Hessian group has shape $(0,0)$. |
| `check_diag_est()`, `set_low_mem(switch=True)` | Boolean for the diagonal diffusion conditions, and the learner with its gradient memory option set, respectively. |
| `update_aux`, `update_aux2` | Update cached numerical arrays in place and return `None`. The grouped method accepts `force_recompute=True`. |

| Regularized estimation | Return value |
| --- | --- |
| `fit(...)` | The fitted learner. `penalty` has $K$ entries and `est_path` has shape $(K,p)$ in ascending penalty order. `val_loss` has shape $(K-2,)$ for an information criterion or holdout, and $(K-2,\texttt{nfolds})$ for expanding windows. |
| `loss(param, penalty=1)`, `gradient(param)` | Scalar penalized objective and gradient vector of the smooth part, respectively. |
| `coef(penalty)`, `aic(k=1)` | Named coefficient dictionary at an exact grid value, and vector of information criteria for the interior grid. |
| `ini_est`, `ini_hess`, `vcov` | `ini_est` and `ini_hess` hold the initial estimate and its symmetrized or adjusted Hessian when used. `vcov` reports the covariance of the base learner, or `None` if unavailable. |
| `lambda_max`, `lambda_maxBW`, `lambda_maxFW` | Upper endpoints for Lasso/Elastic Net, adaptive backward Bridge and quadratic forward Bridge, respectively, when defined. |
| `proximal_gradient(x0, penalty, ...)` | Dictionary with coefficients `x`, objective `f` at those coefficients and penalty, `status`, `message`, update count `niter`, last smooth gradient `jac`, `epsilon` and `stepsize`. `jac` can refer to an extrapolated point or active block. For an interior grid index $i$, `path_info[i-1]` stores the optimizer result for `penalty[i]`. |
| `soft_threshold`, `hard_threshold`, `prox_backtrack` | Vector of updated active coefficients for the corresponding penalty, or a scalar step size for backtracking. |
| `plot(save_fig=None)` | Plots or saves the coefficient paths and returns the learner. |

For custom learners, `SdeLearner` supplies array-to-dictionary adapters `loss_wrap`, `grad_wrap` and `hess_wrap` returning the corresponding scalar, vector and matrix. QMLE also supplies `loss_wrap2` and `grad_wrap2` for group optimization with the other group held fixed. `set_sde` reinitializes a compatible learner in place. The experimental `generate_batch` returns increment indices, `backtrack_rule` returns a step size, and `nesterov_descent` returns the optimizer result dictionary (`x`, `f`, `jac`, `status`, `message`, `niter`, `epsilon` and `batch`).

[^two-step]: Uchida, M. and Yoshida, N. (2012). [Adaptive estimation of an ergodic diffusion process based on sampled data](https://doi.org/10.1016/j.spa.2012.04.001). *Stochastic Processes and their Applications*, 122(8), 2885–2924.

[^elastic-net]: De Gregorio, A., Frisardi, D., Iacus, S. and Iafrate, F. (2025). [Adaptive Elastic-Net estimation for sparse diffusion processes](https://doi.org/10.1007/s11203-025-09341-w). *Statistical Inference for Stochastic Processes*, 28(3), article 22.

[^bridge]: De Gregorio, A. and Iafrate, F. (2021). [Regularized bridge-type estimation with multiple penalties](https://doi.org/10.1007/s10463-020-00769-w). *Annals of the Institute of Statistical Mathematics*, 73(5), 921–951.

[^pathwise]: De Gregorio, A. and Iafrate, F. (2026). [Pathwise optimization for bridge-type estimators and its applications](https://doi.org/10.1007/s11222-026-10856-2). *Statistics and Computing*, 36(3), article 108.
