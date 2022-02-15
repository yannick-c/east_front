" Script which produces an image or constraint(neighbour) and the front progression "
import numpy as np
# import matplotlib.pyplot as plt
import imageio

from tqdm import tqdm

i=0
N = 6  # side length of lattice
vertex_num = np.power(N,2)
q = 0.3  # density of vacancies
MAXT = vertex_num*np.power(2, np.power(np.log2(1/q),
    2)/4+np.log2(1/q))
plot_intervals = np.floor(MAXT/100)

VACANCY = 0
TOUCHED = 125
PARTICLE = 255

grid = np.full((N, N), PARTICLE, dtype=np.uint8)
can_flip = [(0,0)]

# are there vacancies that we can update in the grid?
updatable_vacancies = 0
def constraint(vertex):
    """  Check whether min b.c. East constraints are fulfilled """
    if vertex == (0,0):
        return 1

    for neighbour in [(vertex[0], vertex[1]-1), (vertex[0]-1, vertex[1])]:
        if np.min(neighbour) < 0:
            # don't add vertices outside the lattice
            continue

        if grid[neighbour] == 1:
            return 1

    return 0

def add_ne_neighbours(vertex):
    """ Get the two north east neighbours associated to vertex """
    global updatable_vacancies
    global grid
    for neighbour in [(vertex[0], vertex[1]+1), (vertex[0]+1, vertex[1])]:
        if np.max(neighbour) >= N:
            # don't add vertices outside the lattice
            continue
        if neighbour not in can_flip:
            if grid[neighbour] == VACANCY:
                # could not flip before and is a vacancy
                updatable_vacancies += 1
            can_flip.append(neighbour)

def check_can_flip(vertex):
    """ check whether the North East neighbours of vertex can still flip """
    global updatable_vacancies
    global grid
    for neighbour in [(vertex[0], vertex[1]+1), (vertex[0]+1, vertex[1])]:
        if np.max(neighbour) >= N:
            # don't add vertices outside the lattice
            continue
        if not constraint(neighbour) and neighbour in can_flip:
            if grid[neighbour] == VACANCY:
                # could flip before and is a vacancy
                updatable_vacancies -= 1
            can_flip.remove(neighbour)

curtime = 0
last_plot = plot_intervals
with tqdm(total=MAXT, mininterval=1) as pbar:
    while curtime < MAXT:
        # draw a sample when the next vertex from can_flip is touched
        can_flip_num = len(can_flip)

        if updatable_vacancies:
            next_time = np.random.exponential(1/can_flip_num)
            coin = np.random.random()
        else:
            # we need a new vacancy, so geometrically wait for it
            next_success = np.random.geometric(q)
            next_time = np.random.gamma(next_success, 1/can_flip_num)
            coin = 0 # coin lands on q-probability side

        # the rings are the minimum over all exponential clocks, the minimum of
        # exponential random variable is a single exponential random variable
        # with parameter the sum of all the parameters
        # 
        # A sum of exponential random variables has a gamma distribution with
        # integer shape given by how many variables we sum, we sum a total of
        # next_success random variables with vertex_num parameter (or
        # 1/vertex_num since numpy takes the scale and not the rate)
        curtime = curtime + next_time
        pbar.update(next_time)

        # now take a random choice from the set of vertices that can flip
        ringed_vert = can_flip[np.random.choice(can_flip_num)]
        cur_state = grid[ringed_vert]

        if coin < q and cur_state != VACANCY:
            # was not a vacancy is now a vacancy that can flip
            updatable_vacancies += 1
            grid[ringed_vert] = VACANCY
            add_ne_neighbours(ringed_vert) # add the neighbours to can_flip

        elif coin >= q and cur_state == VACANCY:
            # we put a particle on a vacancy that was updatable
            updatable_vacancies -= 1
            # the hitting time is the firs time you go to the good state so check
            # that you were good before marking this one as 'hit' by
            # putting 0.5 instead of 0
            grid[ringed_vert] = TOUCHED
            check_can_flip(ringed_vert) # remove the neighbours from can_flip
                                        # if they can't flip anymore


        if True:
            last_plot = last_plot + plot_intervals
            # plt.imshow(grid, cmap='Greys', origin="lower",interpolation='none')
            # plt.axis('off')
            # plt.tick_params(axis='both', left='off', top='off', right='off',
            #                 bottom='off', labelleft='off', labeltop='off',
            #                 labelright='off', labelbottom='off')
            # figure = plt.gcf() # get current figure
            # plt.savefig(f"archive2/{int(np.floor(curtime/plot_intervals)):04d}.png", dpi=100,
            #             pad_inches=0, bbox_inches='tight',
            #             transparent=True)
            imageio.imwrite(f"archive/{i}.png", np.rot90(grid))
            i += 1
