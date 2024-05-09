import argparse
from source.system import System
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class Simulation:
    def __init__(self) -> None:
        self.g = np.zeros(2)
        self.system = None
        self.y0 = None
        self.t0 = 0
        self.tf = 1

    def setup(self, settings_file: str) -> None:
        filename = './settings/' + settings_file
        SETTINGS = ''
        with open(filename) as f:
            for line in f.readlines():
                SETTINGS += line.strip() + ' '

        parser = argparse.ArgumentParser()
        parser.add_argument('-N_bodies', dest='N_bodies', type=int, required=True)
        parser.add_argument('-gx', dest='gx', type=float, default=0.)
        parser.add_argument('-gy', dest='gy', type=float, default=-9.81)
        parser.add_argument('-beams', dest='beams', type=str, nargs='+')
        parser.add_argument('-t0', dest='t0', type=float, default=0.)
        parser.add_argument('-tf', dest='tf', type=float, default=1.)
        args = parser.parse_known_args(SETTINGS.split())[0]

        self.g[0] = args.gx
        self.g[1] = args.gy
        self.t0 = args.t0
        self.tf = args.tf

        self.system = System(args.N_bodies, self.g)

        self.y0 = np.zeros(2 * args.N_bodies)

        beam_parser = argparse.ArgumentParser()
        beam_parser.add_argument('-parent', dest='parent', type=int, required=True)
        beam_parser.add_argument('-N_sections', dest='N_sect', type=int, required=True)
        beam_parser.add_argument('-m', dest='m', type=float, default=50.)
        beam_parser.add_argument('-Iz', dest='Iz', type=float, default=50.)
        beam_parser.add_argument('-l', dest='l', type=float, default=2.)
        beam_parser.add_argument('-fixed', dest='fixed', type=bool, default=False)
        beam_parser.add_argument('-dik', dest='dik', type=float, default=[0., 0.], nargs=2)
        beam_parser.add_argument('-pos', dest='pos', type=float, default=[0., 0.], nargs=2)
        beam_parser.add_argument('-angle', dest='angle', type=float, default=0.)
        for f in args.beams:
            filename = './bodies/' + f
            BEAM_SETTINGS = ''
            with open(filename) as file:
                for line in file.readlines():
                    BEAM_SETTINGS += line.strip() + ' '
            args = beam_parser.parse_known_args(BEAM_SETTINGS.split())[0]
            posx_ind, posy_ind, angle_ind = self.system.make_beam(args.N_sect, args.parent, args.l, args.m, args.Iz, args.fixed, np.array(args.dik))
            self.y0[posx_ind] = args.pos[0]
            self.y0[posy_ind] = args.pos[1]
            self.y0[angle_ind] = args.angle

    def run(self):
        sol = solve_ivp(self.system.RHS, (self.t0, self.tf), self.y0, method='DOP853')

        fig, ax = plt.subplots()
        ax.plot(sol.t, sol.y[0], 'k-', label='y[0]')
        ax.plot(sol.t, sol.y[3], 'r--', label='y[3]')
        ax.plot(sol.t, sol.y[4], 'g-.', label='y[4]')
        ax.plot(sol.t, sol.y[5], 'b-', label='y[5]')
        ax.plot(sol.t, sol.y[6], 'y--', label='y[6]')
        #ax.plot(sol.t, sol.y[7], 'b--', label='y[7]')
        #ax.plot(sol.t, sol.y[8], 'y-', label='y[8]')
        ax.grid()
        ax.legend()
        plt.show()
