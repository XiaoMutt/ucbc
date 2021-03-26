from experiments.basis import *


class NormalExperimentWithin(Experiment):
    TITLE = "Normal (Rewards within [0,1])"

    def __init__(self, solvers: tp.List[tp.Type[Solver]]):
        super(NormalExperimentWithin, self).__init__(
            solvers=solvers,
            bandit=NormalBandit([(0.4, 0.1), (0.5, 0.2), (0.6, 0.3)])
        )


class NormalExperimentBeyond(Experiment):
    TITLE = "Normal (Rewards beyond [0,1])"

    def __init__(self, solvers: tp.List[tp.Type[Solver]]):
        super(NormalExperimentBeyond, self).__init__(
            solvers=solvers,
            bandit=NormalBandit([(4, 1), (5, 2), (6, 3)])
        )


class NormalExperimentWithin10RandomArms(Experiment):
    TITLE = "Normal (Rewards within [0,1] 10 random arms)"

    def __init__(self, solvers: tp.List[tp.Type[Solver]]):
        super(NormalExperimentWithin10RandomArms, self).__init__(
            solvers=solvers,
            bandit=NormalBandit([
                [mu, np.random.uniform(0, min(mu, 1 - mu) / 2)]
                for mu in np.random.uniform(0.2, 0.8, 10)
            ])
        )


class NormalExperimentWithin10AdversarialArms(Experiment):
    TITLE = "Normal (Rewards within [0,1] 10 adversarial arms)"

    def __init__(self, solvers: tp.List[tp.Type[Solver]]):
        super(NormalExperimentWithin10AdversarialArms, self).__init__(
            solvers=solvers,
            bandit=NormalBandit([
                [mu, sigma]
                for mu, sigma in zip(np.arange(0.46, 0.55, 0.01), np.arange(0.22, 0.12, -0.01))
            ])
        )
