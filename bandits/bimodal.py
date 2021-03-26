import numpy as np
from .basis import Bandit


class BimodalBandit(Bandit):
    def __init__(self, thetas):
        """
        Setup a Bimodal-arm bandit
        :param thetas: a list of arm parameters. Each parameter is a tuple:
        ((normal1Mean, normal1Variance),(normal2Mean, normal2Variance), probabilityOfNormal1)
        """
        self.thetas = thetas
        self._expects = [theta1[0] * p + theta2[0] * (1 - p) for theta1, theta2, p in self.thetas]
        self._narms = len(self.thetas)
        self._qstar = max(self._expects)

    @classmethod
    def bimodal(cls, theta1, theta2, p):
        if np.random.random() < p:
            return np.random.normal(*theta1)
        else:
            return np.random.normal(*theta2)

    def reward(self, action):
        return self.bimodal(*self.thetas[action])

    def regret(self, action):
        return self.Qstar - self._expects[action]
