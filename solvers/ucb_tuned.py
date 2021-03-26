from .basis import Solver
import numpy as np


class UcbTuned(Solver):
    NAME = "UCB-tuned"
    def __init__(self, bandit):
        super(UcbTuned, self).__init__(bandit)
        self.t = 0
        self.ns = [0] * bandit.narms
        self.sample_means = [0] * bandit.narms
        self.sample_vars = [0] * bandit.narms

        for _ in range(3):
            for a in range(bandit.narms):
                self.update(a)

    def get_upper_bound(self, action):
        tmp = self.sample_means[action] + \
              np.sqrt(
                  (2 * self.sample_vars[action] * np.log(4 * (self.t ** 3))) /
                  self.ns[action]
              ) + \
              (16 * np.log(4 * (self.t ** 3)) / (3 * self.ns[action]))
        return min(tmp, 1)

    def sample(self):
        bounds = [self.get_upper_bound(action) for action in range(self.bandit.narms)]
        # print(bounds, self.ns, self.means)
        return self.argmax(bounds)

    def update(self, action):
        self.t += 1
        reward = self.bandit.reward(action)
        n_ = self.ns[action]
        m_ = self.sample_means[action]
        n = n_ + 1
        m = m_ + (reward - m_) / n
        var_ = self.sample_vars[action]

        var = ((n - 2) / n_ * var_ + (reward - m_) ** 2 / n) if n_ > 0 else 1
        # var = max(2, var)
        var = 1 if var == 0 else 1
        self.ns[action] = n
        self.sample_means[action] = m
        self.sample_vars[action] = var
