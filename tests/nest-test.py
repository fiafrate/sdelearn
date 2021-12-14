import numpy as np
# f: function
# x: point at which to evaluate
# jac_x: gradient at point x
# alpha, gamma: in 0,1
def backtrack_rule(f, x, jac_x, alpha=0.5, gamma=0.8, **kwargs):
    s = 1
    if f(x - s * jac_x, **kwargs) <= f(x, **kwargs) - alpha * s * np.linalg.norm(jac_x) ** 2:
        return s

    while f(x - s * jac_x, **kwargs) > f(x, **kwargs) - alpha * s * np.linalg.norm(jac_x) ** 2:
        s = gamma * s

    return s


def nesterov_descent(f, x0, jac=None, epsilon=1e-03, max_it=1e3, **kwargs):
    x_prev = x0
    t_prev = 1
    y_prev = x_prev
    x_curr = x_prev

    jac_y = jac(y_prev, **kwargs)

    it_count = 1
    rel_err = 1

    while np.linalg.norm(jac_y) >= epsilon and it_count < max_it:

        s = backtrack_rule(f, y_prev, jac_y, alpha=0.5, gamma=0.8, **kwargs)
        print(s)

        x_curr = y_prev - s * jac_y

        # interrupt execution before costly evaluation of the gradient
        # if np.linalg.norm(x_curr - x_prev)/np.linalg.norm(x_prev) < epsilon:
        #     message = 'relative error termination reached'
        #     print(message)
        #     return x_curr

        t_curr = (1 + np.sqrt(1 + 4 * t_prev ** 2)) / 2

        y_curr = x_curr + (t_prev - 1) / t_curr * (x_curr - x_prev)

        x_prev = x_curr
        t_prev = t_curr
        y_prev = y_curr
        #
        jac_y = jac(y_curr, **kwargs)
        print(x_curr)
        print(jac_y)

        it_count += 1
    return x_curr


def f(x):
    return x*x


def jac(x):
    return 2*x

nesterov_descent(f, 10, jac)

from ctest import test

test(5)