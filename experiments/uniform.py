from experiments.basis import *


class UniformExperimentWithin(Experiment):
    TITLE = "Uniform (Rewards within [0,1])"

    def __init__(self, solvers: tp.List[tp.Type[Solver]]):
        super(UniformExperimentWithin, self).__init__(
            solvers=solvers,
            bandit=UniformBandit([(0, 0.4), (0, .5), (0, 0.6)])
        )


class UniformExperimentBeyond(Experiment):
    TITLE = "Uniform (Rewards beyond [0,1])"

    def __init__(self, solvers: tp.List[tp.Type[Solver]]):
        super(UniformExperimentBeyond, self).__init__(
            solvers=solvers,
            bandit=UniformBandit([(0, 4), (0, 5), (0, 6)])
        )


class UniformExperimentWithin10RandomArms(Experiment):
    TITLE = "Uniform (Rewards Within [0,1] 10 random arms)"

    def __init__(self, solvers: tp.List[tp.Type[Solver]]):
        super(UniformExperimentWithin10RandomArms, self).__init__(
            solvers=solvers,
            bandit=UniformBandit([sorted(np.random.random(2)) for _ in range(10)])
        )
