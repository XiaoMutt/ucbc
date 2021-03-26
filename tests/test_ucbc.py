import numpy as np
from bandits import BernoulliBandit
from solvers import Ucbc
from collections import defaultdict
from tqdm import tqdm
from unittest import TestCase


class UcbcWrapper(Ucbc):
    def __init__(self, bandit):
        self.rewards = defaultdict(list)
        super(UcbcWrapper, self).__init__(bandit)

    def update(self, action):
        reward = super(UcbcWrapper, self).update(action)
        self.rewards[action].append(reward)
        return reward


class TestUcbc(TestCase):
    def setUp(self) -> None:
        self.bandit = BernoulliBandit([0, 0.1, 0.3, 0.5, 0.6, 0.9])
        self.vars = [theta * (1 - theta) for theta in self.bandit.thetas]
        self.solver = UcbcWrapper(bandit=self.bandit)

    def test_sample_variance(self):
        for _ in tqdm(range(1000000)):
            action = self.solver.sample()
            self.solver.update(action)

        for action, values in self.solver.rewards.items():
            self.assertAlmostEqual(np.var(values, ddof=1), self.solver.sample_vars[action])

