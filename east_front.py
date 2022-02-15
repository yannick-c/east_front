"""
Script that simulates the 2D East model on Z^2_+ with minimal boundary
conditions and gives images.
"""

import os
import multiprocessing
import time
from typing import Tuple, List, Any
import sys

import numpy as np  # type: ignore
import imageio  # type: ignore

from tqdm import tqdm  # type: ignore


DEBUG = False

VACANCY = 0
TOUCHED = 125
PARTICLE = 255

class EastChain:
    """
    Class containing the whole descritpion of the East chain and its
    evolution starting from the filled state with minimal boundary conditions.
    """

    def __init__(self, side_length = 1000, density = 0.05, two_dim_time = True,
                 plot_interval_percentage = 0.1):
        # All the parameters of the model

        # side length of the box (number of vertices)
        self.side_length = side_length

        # density of vacancies in equilibrium
        self.q = density # pylint: disable=invalid-name


        # if True stop the simulation after N/velocity
        # of two-dimensional bulk, otherwise take one-dimensional time
        self.two_dim_time = two_dim_time

        # After how many percents of the simulation you should save a plot
        self.plot_interval_percentage = plot_interval_percentage

        # the starting configuration is always filled
        self.configuration = np.full((self.side_length, self.side_length),
                                      PARTICLE, dtype=np.uint8)

        self.can_flip, self.updatable_vacancies = self.analyse_initial_state()


    def analyse_initial_state(self) -> Tuple[List[Tuple[int, int]], int]:
        """
        Find the initial states that can flip and are updatable vacancies.
        """

        can_flip = []
        updatable_vacancies = 0

        if not np.any(self.configuration - PARTICLE):
            # there are only particles on the configuration
            # so only check the sides
            can_flip.append((0,0))

            for vertex_side in range(1, self.side_length):
                for vertex in [(vertex_side, 0), (0, vertex_side)]:
                    if self.constraint(vertex):
                        can_flip.append(vertex)
                        if self.configuration[vertex] == VACANCY:
                            updatable_vacancies += 1

        else:
            # check everything
            for vertex_0 in range(self.side_length):
                for vertex_1 in range(self.side_length):

                    vertex = (vertex_0, vertex_1)

                    if self.constraint(vertex):
                        can_flip.append(vertex)
                        if self.configuration[vertex] == VACANCY:
                            updatable_vacancies += 1

        return can_flip, updatable_vacancies

    def ne_neighbours(self, vertex: Tuple[int, int],
                      one_dimensional = False) -> List[Tuple[int, int]]:
        """
        Return the north-east neighbours that are contained in the grid
        """

        if one_dimensional:
            neighbours = [nb for nb in [(vertex[0]+1, vertex[1]),
                                         (vertex[0], vertex[1]+1)]
                          if np.max(nb) < self.side_length
                          and np.min(nb) == 0]

        else:
            neighbours = [nb for nb in [(vertex[0]+1, vertex[1]),
                                         (vertex[0], vertex[1]+1)]
                          if np.max(nb) < self.side_length]

        return neighbours

    def sw_neighbours(self, vertex: Tuple[int, int]) -> List[Tuple[int, int]]: # pylint: disable=no-self-use
        """
        Return the south-west neighbours that are contained in the grid
        """

        neighbours = [nb for nb in [(vertex[0]-1, vertex[1]),
                                    (vertex[0], vertex[1]-1)]
                      if np.min(nb) >= 0]
        return neighbours

    def constraint(self, vertex: Tuple[int, int]) -> int:
        """
        This checks whether the vertex is unconstrained in the current
        configuration, returns 1 if this is the case, 0 otherwise.
        """

        # maximal boundary conditions:
        # if np.min(vertex) == 0:
        #     return 1

        # minimal boundary conditions
        if vertex == (0,0):
            return 1

        if DEBUG:
            goodnb = [nb for nb in self.sw_neighbours(vertex)
                      if self.configuration[nb] == VACANCY]
            print(f"Constraint of {vertex} fulfiled by: {goodnb}")

        if [nb for nb in self.sw_neighbours(vertex)
                if self.configuration[nb] == VACANCY]:
            return 1

        return 0

    def ne_neighbours_unconstrain(self, vertex: Tuple[int, int],
                                  one_dimensional=False):
        """
        In the context in which vertex has just been updated from PARTICLE to
        VACANCY add the north-east neighbours of vertex to self.can_flip if they
        are not already contained.
        """

        for neighbour in set(self.ne_neighbours(vertex, one_dimensional)).difference(self.can_flip):
            self.can_flip.append(neighbour)

            if self.configuration[neighbour] == VACANCY:
                self.updatable_vacancies += 1

    def ne_neighbours_constrain(self, vertex: Tuple[int, int],
                                one_dimensional=False):
        """
        In the context in which vertex has just been updated from VACANCY to
        PARTICLE check the constraints for the north-east neighbours of vertex
        and if not fulfilled remove them from can_flip (no need to check since
        vertex was VACANCY before so the vertex should be in can_flip).
        """

        for neighbour in [nb for nb in self.ne_neighbours(vertex, one_dimensional)
                          if not self.constraint(nb)]:
            self.can_flip.remove(neighbour)
            if self.configuration[neighbour] == VACANCY:
                self.updatable_vacancies -= 1

    def step(self, one_dimensional=False) -> float:
        """
        Make a single update step using all the optimizations and return a
        float with the time that has passed in this step.
        """

        # distinguish two cases, the one where we have updatable vacancies
        # and the one where we do not.

        if self.updatable_vacancies > 0:
            # in this case we wait for a ring on the unconstrained states
            # (the rings outside do not change anything anyway)
            elapsed = np.random.exponential(1/len(self.can_flip))

            # take a random coin to choose between putting 0 or 1
            coin = np.random.random()

            if DEBUG:
                print(f"normal ring, time {elapsed} with coin {coin}")

        else:
            # in this case we do not take a random coin but wait for the
            # next flip that puts a vacancy since every other flip does not
            # change anything. The chance to get a q flip on a coin is
            # geometric
            next_q_flip = np.random.geometric(self.q)

            # the sum of k i.i.d. exponential random variables is the gamma
            # distribution
            elapsed = np.random.gamma(next_q_flip, 1/len(self.can_flip))

            # we ensure a q flip so put the coin to 0
            coin = 0

            if DEBUG:
                print(f"wait for q ring, time {elapsed} with coin {coin}")


        ringed_vert = self.can_flip[np.random.choice(len(self.can_flip))]

        current_state = self.configuration[ringed_vert]

        if DEBUG:
            print(f"ring at: {ringed_vert} in state {current_state}")

        if coin <= self.q and current_state != VACANCY:
            self.updatable_vacancies += 1
            self.configuration[ringed_vert] = VACANCY
            self.ne_neighbours_unconstrain(ringed_vert, one_dimensional)

        elif coin > self.q and current_state == VACANCY:
            self.updatable_vacancies -= 1
            self.configuration[ringed_vert] = TOUCHED
            self.ne_neighbours_constrain(ringed_vert, one_dimensional)

        if DEBUG:
            print(f"updatable_vacancies: {self.updatable_vacancies}\n"
                  f"configuration:\n {self.configuration}\n"
                  f"can_flip: {self.can_flip}")
            time.sleep(1)

        return elapsed

    def plot(self, fname: Any):
        """ Plot the current configuration as a greyscale matrix """
        imageio.imwrite(f"archive_q004_l1000/{fname:08d}.png", np.rot90(self.configuration))

    def simulate_plot(self):
        """ Simulate and plot the evolution of the configuration """

        theta = np.log2(1/self.q)

        if self.two_dim_time:
            max_t = self.side_length * np.power(2, np.power(theta , 2)/4 +
                                                theta*np.log2(theta))
        else:
            max_t = 100*self.side_length * np.power(2, np.power(theta , 2)/2 +
                                                    theta*np.log2(theta))

        plot_intervals = np.ceil(max_t*self.plot_interval_percentage/100)

        curtime = 0
        next_plot = plot_intervals
        with tqdm(total=max_t, mininterval=1) as pbar:
            updater = pbar.update
            while curtime < max_t:
                elapsed = self.step()
                updater(elapsed)
                curtime += elapsed
                if curtime > next_plot:
                    next_plot += plot_intervals
                    fname = int(np.floor(curtime/plot_intervals))
                    self.plot(fname)

    def remove_non_1d(self):
        """ Remove any entry in self.can_flip that is not on the border.  """
        new_can_flip = []
        for vertex in self.can_flip:
            if np.min(vertex) > 0:
                # not on the border
                if self.configuration[vertex] == VACANCY:
                    self.updatable_vacancies -= 1
            else:
                new_can_flip.append(vertex)

        self.can_flip = new_can_flip

    def simulate_hitting(self):
        """
        Simulate the chain and return the hitting time of the top-right
        corner and the minimum of the top left and bottom right corner (i.e.
        two-dimensional times).

        num_runs is an integer indicating how many total runs should be done
        for the average.
        """
        curtime = 0

        bulk_hitting = 0

        edge = self.side_length - 1


        while True:
            curtime += self.step()

            if (not bulk_hitting
                    and self.configuration[edge, edge] == VACANCY):
                bulk_hitting = curtime
                print(f"\nset bulk time = {curtime}")
                break

        print("\nGo one dimensional")
        # cull the can flip list to only include the borders
        self.remove_non_1d()

        while True:
            curtime += self.step(one_dimensional=True)

            if VACANCY in (self.configuration[edge, 0],
                           self.configuration[0, edge]):
                print(f"\nhit axis at {curtime}")
                return bulk_hitting, curtime


def init_for_hitting(**kwargs):
    """
    kwargs are passed on to init of EastChain.
    """
    # seed the random number generator different for each child
    np.random.seed(int.from_bytes(os.urandom(4),
                                  byteorder='little'))

    return EastChain(**kwargs).simulate_hitting()

def get_hitting_time(num_runs: int, **kwargs):
    """
    Multiprocess the hitting time getting.

    kwargs are passed to EastChain.
    """

    times = []

    if DEBUG:
        proc_num = 1
    else:
        proc_num = 5

    with tqdm(total=num_runs) as pbar:

        def update_bar(_):
            pbar.update(1)

        with multiprocessing.Pool(proc_num) as pool:
            results = [pool.apply_async(init_for_hitting, kwds=kwargs,
                                        callback=update_bar)
                       for _ in range(num_runs)]

            for result in results:
                times.append(result.get())

    bulk_times, axis_times = zip(*times)

    print(f"\nWe get the average bulk_times = {np.min(bulk_times)},"
          f" {np.average(bulk_times)}, {np.max(bulk_times)} and"
          f" axis_times = {np.min(axis_times)}, {np.average(axis_times)},"
          f" {np.max(axis_times)} on a run of {num_runs}"
          " iterations.")


if __name__ == "__main__":
    EastChain(side_length=1000, density=0.04, plot_interval_percentage=0.005).simulate_plot()
    # get_hitting_time(100, density=0.025, side_length=100)
