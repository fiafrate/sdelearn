import numpy as np
import copy


class SdeSampling:
    """
    A class to handle sampling for stochastic differential equations (SDEs).
    Can create a grid of times between an initial and terminal point.
    Offers sub-sampling functionality.

    - `initial`: The initial time.
    - `terminal`: The terminal time.
    - `n`: Number of points in the grid.
    - `delta`: The difference between consecutive points in the grid.
    - `grid`: An array representing the grid of times.
    - `x0`: Initial value for a potential SDE, defaulted to `None`.

    """
    def __init__(self, initial, terminal, n=None, delta=None):
        """
        Construct a sampling object.
        :param initial: initial sampling time
        :param terminal: final sampling time
        :param n: number of points in the grid
        :param delta: The difference between consecutive points in the grid.

        `initial` and `terminal` parameters are required as well as only one between
        `n` and `delta`.
        """
        self.initial = initial

        assert n is not None or delta is not None, "n or delta parameter missing"

        if n is not None:
            self.grid, self.delta = np.linspace(initial, terminal, n, retstep=True)
            self.n = n
            self.terminal = terminal
        if delta is not None:
            self.delta = delta
            self.grid = np.arange(initial, terminal + delta, delta)
            self.grid = self.grid[self.grid <= terminal]
            self.n = len(self.grid)
            self.terminal = self.grid[self.n - 1]
        self.x0 = None

    def sub_sampling(self, first_n=None, last_n=None, new_term=None, new_init=None, from_range=None):
        """
        return a new sampling object obtained as a subset of the current time indices.
        Specify *only one* of the following arguments (if one wants both new init and new terminal call the function
        twice, e.g. by concatenating the calls )
        :param first_n: int, take the first n elements
        :param last_n: int, take the last n elements
        :param new_term: float, new terminal value (if not an exact, the closest value in the grid will be chosen)
        :param new_init: float, new initial value, as above
        :param from_range: array-like of length 2: giving the first (included) and last (not included) indices of times
            to include
        :return: new sampling object. If no parameter is given returns a copy of the original one
        """
        new_samp = copy.deepcopy(self)

        if first_n is not None:
            new_samp.grid = new_samp.grid[:first_n]
            new_samp.terminal = new_samp.grid[first_n - 1]
            new_samp.n = first_n
        elif last_n is not None:
            new_samp.grid = new_samp.grid[-last_n:]
            new_samp.initial = new_samp.grid[-last_n]
            new_samp.n = last_n
            new_samp.x0 = None
        elif new_term is not None:
            new_samp.grid = new_samp.grid[:np.argmin(np.abs(new_samp.grid - new_term))]
            new_samp.terminal = new_samp.grid[- 1]
            new_samp.n = len(new_samp.grid)
        elif new_init is not None:
            new_samp.grid = new_samp.grid[np.argmin(np.abs(new_samp.grid - new_init)):]
            new_samp.initial = new_samp.grid[0]
            new_samp.n = len(new_samp.grid)
            new_samp.x0 = None
        elif from_range is not None:
            new_samp.grid = new_samp.grid[from_range[0]:from_range[1]]
            new_samp.initial = new_samp.grid[0]
            new_samp.terminal = new_samp.grid[- 1]
            new_samp.n = len(new_samp.grid)
            new_samp.x0 = None
        return new_samp

    def __str__(self):
        out = '\nSde Sampling object ---- \n\nInitial time: {0}, Terminal time: {1}\n{2} observations with time delta {3}'.format(
            self.initial, self.terminal, self.n, self.delta)
        return out
