from .basis import Solver
import numpy as np


class Ucb1(Solver):
    NAME = "UCB1"

    def __init__(self, bandit):
        super(Ucb1, self).__init__(bandit)
        self.t = 0
        self.ns = [0] * bandit.narms
        self.means = [0.] * bandit.narms

        for action in range(self.bandit.narms):
            self.update(action)

    def get_upper_bound(self, action):
        return self.means[action] + np.sqrt(
            np.log(self.t) /
            self.ns[action]
        )

    def sample(self):
        bounds = [self.get_upper_bound(action) for action in range(self.bandit.narms)]
        return self.argmax(bounds)

    def update(self, action):
        reward = self.bandit.reward(action)
        self.t += 1
        self.ns[action] += 1
        self.means[action] += (reward - self.means[action]) / self.ns[action]
        return reward

