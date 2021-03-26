from experiments.basis import *


class BimodalExperimentWithin(Experiment):
    TITLE = "Bimodal (Rewards within [0,1])"

    def __init__(self, solvers: tp.List[tp.Type[Solver]]):
        super(BimodalExperimentWithin, self).__init__(
            solvers=solvers,
            bandit=BimodalBandit([
                ((0.4, 0.1), (0.6, 0.3), 0.4),
                ((0.5, 0.2), (0.5, 0.2), 0.5),
                ((0.6, 0.3), (0.3, 0.1), 0.6)
            ])
        )


class BimodalExperimentBeyond(Experiment):
    TITLE = "Bimodal (Rewards beyond [0,1])"

    def __init__(self, solvers: tp.List[tp.Type[Solver]]):
        super(BimodalExperimentBeyond, self).__init__(
            solvers=solvers,
            bandit=BimodalBandit([
                ((4, 1), (6, 3), 0.4),
                ((5, 2), (5, 2), 0.5),
                ((6, 3), (3, 1), 0.6)
            ])
        )


class BimodalExperimentWithin10RandomArms(Experiment):
    TITLE = "Bimodal (Rewards within [0,1] 10 random arms)"

    def __init__(self, solvers: tp.List[tp.Type[Solver]]):
        super(BimodalExperimentWithin10RandomArms, self).__init__(
            solvers=solvers,
            bandit=BimodalBandit([
                ((mu1, np.random.uniform(0, min(mu1, 1 - mu1) / 2)),
                 (mu2, np.random.uniform(0, min(mu1, 1 - mu1) / 2)),
                 p)
                for mu1, mu2, p in zip(np.random.uniform(0.1, 0.7, 10),
                                       np.random.uniform(0.3, 0.9, 10),
                                       np.random.uniform(0, 1, 10))

            ])
        )
