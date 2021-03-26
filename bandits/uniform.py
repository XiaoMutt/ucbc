from .basis import Bandit
import numpy as np


class UniformBandit(Bandit):
    def __init__(self, thetas):
        self.thetas = thetas
        self.means = [sum(theta) / 2 for theta in self.thetas]
        self._narms = len(self.thetas)
        self._qstar = max(self.means)

    def reward(self, action):
        return np.random.uniform(*self.thetas[action])

    def regret(self, action):
        return self._qstar - self.means[action]
