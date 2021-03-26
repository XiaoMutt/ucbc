import numpy as np
from .basis import Bandit


class BernoulliBandit(Bandit):
    def __init__(self, thetas):
        self.thetas = thetas
        self._narms = len(self.thetas)
        self._qstar = max(self.thetas)

    def reward(self, action):
        return np.random.binomial(1, self.thetas[action])

    def regret(self, action):
        return self.Qstar - self.thetas[action]
