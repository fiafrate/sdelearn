import numpy as np
import pandas as pd


class SdeData:
    def __init__(self, data):
        self.original_data = data
        self.data = pd.DataFrame(data)
        self.n_obs = self.data.shape[0]

    def format_data(self, time_index=None, col_names=None):
        if time_index is not None:
            self.data.set_index(pd.Index(time_index), inplace=True)
        if col_names is not None:
            self.data.columns = col_names

