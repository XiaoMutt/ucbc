from experiments import *
from solvers import *
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    steps = 1000000
    num_episodes = 100
    solvers = [Ucb1, Ucbc]
    experiments = [
        BernoulliExperimentWithin(solvers),
        BernoulliExperimentWithin10RandomArms(solvers),
        BimodalExperimentWithin(solvers),
        BimodalExperimentBeyond(solvers),
        BimodalExperimentWithin10RandomArms(solvers),
        NormalExperimentWithin(solvers),
        NormalExperimentBeyond(solvers),
        NormalExperimentWithin10RandomArms(solvers),
        UniformExperimentWithin(solvers),
        UniformExperimentBeyond(solvers),
        UniformExperimentWithin10RandomArms(solvers),
        BernoulliExperimentWithin10AdversarialArms(solvers),
        NormalExperimentWithin10AdversarialArms(solvers),

    ]

    for exp in experiments:
        print(f"Doing {exp.TITLE}")
        exp(num_steps=steps, num_episodes=num_episodes)
        print(f"Done")
