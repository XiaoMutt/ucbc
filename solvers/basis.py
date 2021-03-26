import numpy as np


class Solver(object):
    NAME = "Solver"

    def __init__(self, bandit):
        self.bandit = bandit

    @classmethod
    def argmax(cls, params):
        # argmax random on tie using reservoir sampling
        res = 0
        maximum = float('-inf')
        k = 0
        for i, p in enumerate(params):
            if p > maximum:
                maximum = p
                k = 1
                res = i
            elif p == maximum:
                k += 1
                if np.random.random() < 1 / k:
                    res = i
        return res

    def sample(self):
        """
        Sample an action
        :return:
        """
        raise NotImplementedError

    def update(self, action) -> float:
        """
        Update parameters based on action
        :param action:
        :return: the reward from the bandit by taking this action
        """
        raise NotImplementedError
