import numpy as np
from .basis import Solver


class ThompsonSampling(Solver):
    NAME = "ThompsonSampling"

    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self.arms = [[1, 1] for _ in range(bandit.narms)]

    def sample(self):
        return self.argmax([np.random.beta(*arm) for arm in self.arms])

    def update(self, action):
        reward = self.bandit.reward(action)
        self.arms[action][int(reward > 0.5) ^ 1] += 1
        return reward
