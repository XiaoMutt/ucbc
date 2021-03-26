from .basis import Solver
import numpy as np


class Ucbc(Solver):
    NAME = "UCBC"

    def __init__(self, bandit):
        super(Ucbc, self).__init__(bandit)
        self.t = 0
        self.ns = [0] * bandit.narms
        self.sample_means = [0] * bandit.narms
        self.sample_vars = [0] * bandit.narms

        for a in range(bandit.narms):
            self.update(a)

    def get_upper_bound(self, action):
        return self.sample_means[action] + np.sqrt(
            (self.get_sample_var(action) * np.log(self.t)) /
            (np.random.chisquare(self.ns[action]))
        )

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
        # var = 1 if var == 0 else var
        self.ns[action] = n
        self.sample_means[action] = m
        self.sample_vars[action] = var
        return reward

    def get_sample_var(self, action):
        v = self.sample_vars[action]
        n = self.ns[action]
        res = (v + 1 / n) / np.tanh(np.log10(n + 1))
        return res
