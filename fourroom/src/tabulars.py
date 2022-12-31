import numpy as np


class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state, ])

    def __len__(self):
        return self.nstates
