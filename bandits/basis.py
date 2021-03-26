class Bandit(object):
    _narms: int
    _qstar: float

    def reward(self, action) -> float:
        raise NotImplementedError

    def regret(self, action) -> float:
        raise NotImplementedError

    @property
    def narms(self):
        return self._narms

    @property
    def Qstar(self):
        return self._qstar
