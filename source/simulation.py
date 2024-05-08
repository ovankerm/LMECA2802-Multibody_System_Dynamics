import argparse
from source.system import System
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class Simulation:
    def __init__(self) -> None:
        self.t = 0
        self.g = np.zeros(2)
        self.system = None

    def setup(self, settings_file: str) -> None:
        filename = './settings/' + settings_file
        SETTINGS = ''
        with open(filename) as f:
            for line in f.readlines():
                SETTINGS += line.strip() + ' '

        parser = argparse.ArgumentParser()
        parser.add_argument('-gx', dest='gx', type=float)
        parser.add_argument('-gy', dest='gy', type=float)
        parser.add_argument('-N_bodies', dest='N_bodies', type=int)
        args = parser.parse_known_args(SETTINGS.split())[0]

        self.g[0] = args.gx
        self.g[1] = args.gy
        self.system = System(args.N_bodies)

        self.system.make_beam(7, 0, fixed=True)

    def run(self):
        sol = solve_ivp(self.system.RHS, (0, 2), np.zeros(2 * 9), method='DOP853')

        fig, ax = plt.subplots()
        ax.plot(sol.t, sol.y[0], 'k-', label='y[0]')
        ax.plot(sol.t, sol.y[3], 'r--', label='y[3]')
        ax.plot(sol.t, sol.y[4], 'g-.', label='y[4]')
        ax.plot(sol.t, sol.y[5], 'b-', label='y[5]')
        ax.plot(sol.t, sol.y[6], 'y--', label='y[6]')
        ax.plot(sol.t, sol.y[7], 'b--', label='y[7]')
        ax.plot(sol.t, sol.y[8], 'y-', label='y[8]')
        ax.grid()
        ax.legend()
        plt.show()
