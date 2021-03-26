from bandits import *
from solvers import *
from multiprocessing import Process, Queue
from threading import Thread
from queue import Empty
from queue import Queue as ThreadQueue
import os
import typing as tp
from tqdm import tqdm
import matplotlib.pyplot as plt
import struct
import numpy as np
import pickle


class Experiment(object):
    COLOR = {
        Ucb1: "k",
        Ucbc: "r",
        ThompsonSampling: "m",
    }

    TITLE = ''

    def __init__(self, solvers: tp.List[tp.Type[Solver]], bandit: Bandit, ncpu: int = 4):
        self.ncpu = ncpu
        self.folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

        # check color
        for solver in solvers:
            if solver not in self.COLOR:
                raise Exception(f"Color of {solver} is not defined")

        self.solvers: tp.List[tp.Type[Solver]] = solvers

        self.bandit: Bandit = bandit
        self.monitor_queue = Queue()  # send job data to monitor from processors
        self.job_queue = Queue()  # the job queue for processors

    @classmethod
    def plotter(cls, solvers, plotter_queue, num_steps, filename):
        # start matplotlib GUI

        figure, axes = plt.subplots(1, len(solvers) + 1, sharex=True, figsize=(13, 4))
        solver2index = {solver: i for i, solver in enumerate(solvers)}
        ax_titles = [solver.NAME for solver in solvers]
        ax_titles.append("Overlay")

        for index, ax in enumerate(axes):
            ax.set_aspect('equal', adjustable='box')
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1, num_steps)
            ax.set_ylim(1, num_steps)
            ax.set_title(ax_titles[index])
            if index == 0:
                ax.set_ylabel('Total Regret')
            ax.set_xlabel('Time')

        figure.suptitle(cls.TITLE)

        while True:
            args = plotter_queue.get()
            if args is None:
                break

            solver, regrets = args
            values = np.cumsum(regrets)
            axes[solver2index[solver]].plot(values, cls.COLOR[solver], alpha=0.05)
            axes[-1].plot(values, cls.COLOR[solver], alpha=0.05)
            plt.pause(0.05)

        if filename:
            # file name is given. save figure
            figure.savefig(filename + ".png", dpi=300)
            plt.cla()
            plt.xscale("linear")  # suppress clf log axes warning
            plt.yscale("linear")  # suppress clf log axes warning
            plt.clf()
            plt.close()
        else:
            plt.show()

    def monitor(self, plotter_queue, filename, num_steps, num_episodes):

        pickled_solvers = pickle.dumps(self.solvers)

        with open(filename + '.dat', 'wb') as writer, tqdm(total=num_episodes * len(self.solvers)) as pbar:
            solver2index = {solver
                            : i for i, solver in enumerate(self.solvers)}
            # save data to file
            # the first 4 bytes are num_steps, and then 4 bytes are the length of pickled solvers data,
            # all as unsigned int
            writer.write(struct.pack('II', num_steps, len(pickled_solvers)))
            # followed by pickled solvers data
            writer.write(pickled_solvers)

            # title length
            title = self.TITLE.encode()
            writer.write(struct.pack('I', len(title)))
            # title
            writer.write(title)
            running_processor = self.ncpu
            while running_processor:
                args = self.monitor_queue.get()
                if args is None:
                    running_processor -= 1
                else:
                    episode, regrets, solver = args

                    # send to plotter
                    plotter_queue.put((solver, regrets))

                    # write to file
                    writer.write(struct.pack('B', solver2index[solver]))  # the solver's index as 1 byte unsigned int
                    values = (regrets * 100).astype('uint16')
                    writer.write(values.tobytes())  # regrets as uint16 * num_steps

                    pbar.set_description(f"episode {episode} of {solver.NAME} ")
                    pbar.update()
            # end plotter
            plotter_queue.put(None)

    def processor(self, num_steps):
        while True:
            try:
                args = self.job_queue.get(timeout=3)
            except Empty:
                break
            episode, solver = args

            s = solver(self.bandit)
            regrets = np.empty(num_steps)

            for i in range(num_steps):
                action = s.sample()
                regret = self.bandit.regret(action)
                regrets[i] = regret
                s.update(action)

            self.monitor_queue.put((episode, regrets, solver))

        # send signal to monitor
        self.monitor_queue.put(None)

    def __call__(self, num_steps: int, num_episodes: int):

        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        filename = os.path.join(self.folder, self.__class__.__name__ + '_' +
                                '_'.join([solver.NAME for solver in self.solvers]) +
                                f"_{num_steps}steps_{num_episodes}episodes")

        for episode in range(1, num_episodes + 1):
            for solver in self.solvers:
                self.job_queue.put((episode, solver))

        plotter_queue = ThreadQueue()
        # start monitor
        monitor = Thread(target=self.monitor, args=(plotter_queue, filename, num_steps, num_episodes))
        monitor.start()

        # start job processor
        processors = []
        for _ in range(self.ncpu):
            processors.append(Process(target=self.processor, args=(num_steps,)))
            processors[-1].start()

        self.plotter(self.solvers, plotter_queue, num_steps, filename)

        # wait for processors to quit
        for p in processors:
            p.join()

    @classmethod
    def _load_helper(cls, reader, plotter_queue, num_steps, solvers):
        while True:
            data = reader.read(1)
            if not data:
                break

            solver_index = struct.unpack('B', data)[0]
            data = reader.read(num_steps * 2)
            regrets = np.frombuffer(data, dtype='uint16')
            plotter_queue.put((solvers[solver_index], regrets / 100))

        plotter_queue.put(None)

    @classmethod
    def load(cls, filepath):
        # start plotter

        with open(filepath, 'rb') as reader:
            num_steps, pickled_solvers_length = struct.unpack('II', reader.read(8))
            solvers = pickle.loads(reader.read(pickled_solvers_length))

            title_length = struct.unpack('I', reader.read(4))[0]
            cls.TITLE = reader.read(title_length).decode()
            plotter_queue = ThreadQueue()
            helper = Thread(target=cls._load_helper, args=(reader, plotter_queue, num_steps, solvers))
            helper.start()
            cls.plotter(solvers, plotter_queue, num_steps, None)
            cls.TITLE = ''

        plt.show()
