from .basis import Bandit
import numpy as np


class NormalBandit(Bandit):
    def __init__(self, thetas: list):
        self.thetas = thetas
        self._narms = len(self.thetas)
        self._qstar = max(theta[0] for theta in self.thetas)

    def reward(self, action):
        return np.random.normal(*self.thetas[action])

    def regret(self, action):
        return self.Qstar - self.thetas[action][0]
