from experiments.basis import *


class BernoulliExperimentWithin(Experiment):
    TITLE = "Bernoulli (Rewards within [0,1])"

    def __init__(self, solvers: tp.List[tp.Type[Solver]]):
        super(BernoulliExperimentWithin, self).__init__(
            solvers=solvers,
            bandit=BernoulliBandit([0.1, 0.5, 0.7])
        )


class BernoulliExperimentWithin10RandomArms(Experiment):
    TITLE = "Bernoulli (Rewards within [0,1] 10 random arms)"

    def __init__(self, solvers: tp.List[tp.Type[Solver]]):
        super(BernoulliExperimentWithin10RandomArms, self).__init__(
            solvers=solvers,
            bandit=BernoulliBandit(np.random.random(10))
        )


class BernoulliExperimentWithin10AdversarialArms(Experiment):
    TITLE = "Bernoulli (Rewards within [0,1] 10 adversarial arms)"

    def __init__(self, solvers: tp.List[tp.Type[Solver]]):
        super(BernoulliExperimentWithin10AdversarialArms, self).__init__(
            solvers=solvers,
            bandit=BernoulliBandit(np.arange(0.46, 0.55, 0.01))
        )
