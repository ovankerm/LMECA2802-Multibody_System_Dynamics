import argparse
from source.system import System
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import glob

class Simulation:
    def __init__(self) -> None:
        self.g = np.zeros(2)
        self.system = None
        self.y0 = None
        self.t0 = 0
        self.tf = 1
        self.name = ''

    def setup(self, settings_file: str, name: str) -> None:
        self.name = name
        savedir = './results/' + self.name + '/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        else:
            for file in glob.glob(savedir + '*'):
                os.remove(file)

        filename = './settings/' + settings_file
        SETTINGS = ''
        with open(filename) as f:
            for line in f.readlines():
                SETTINGS += line.strip() + ' '

        parser = argparse.ArgumentParser(prefix_chars='+')
        parser.add_argument('+N_bodies', dest='N_bodies', type=int, required=True)
        parser.add_argument('+gx', dest='gx', type=float, default=0.)
        parser.add_argument('+gy', dest='gy', type=float, default=-9.81)
        parser.add_argument('+beams', dest='beams', type=str, nargs='+')
        parser.add_argument('+t0', dest='t0', type=float, default=0.)
        parser.add_argument('+tf', dest='tf', type=float, default=1.)
        parser.add_argument('+forces', dest='forces', type=str, nargs='+')
        parser.add_argument('+dis', dest='dis', type=float, nargs='+')
        parser.add_argument('+save_forces', dest='s_force', type=bool, default=False)
        args = parser.parse_known_args(SETTINGS.split())[0]

        self.g[0] = args.gx
        self.g[1] = args.gy
        self.t0 = args.t0
        self.tf = args.tf

        self.system = System(args.N_bodies, self.g, args.s_force, savedir+'forces.txt')

        self.y0 = np.zeros(2 * args.N_bodies)

        beam_parser = argparse.ArgumentParser(prefix_chars='+')
        beam_parser.add_argument('+parent', dest='parent', type=int, required=True)
        beam_parser.add_argument('+N_sections', dest='N_sect', type=int, required=True)
        beam_parser.add_argument('+m', dest='m', type=float, default=50.)
        beam_parser.add_argument('+Iz', dest='Iz', type=float, default=50.)
        beam_parser.add_argument('+l', dest='l', type=float, default=2.)
        beam_parser.add_argument('+fixed', dest='fixed', type=bool, default=False)
        beam_parser.add_argument('+dik', dest='dik', type=float, default=[0., 0.], nargs=2)
        beam_parser.add_argument('+pos', dest='pos', type=float, default=[0., 0.], nargs=2)
        beam_parser.add_argument('+angle', dest='angle', type=float, default=0.)
        beam_parser.add_argument('+bodies_names', dest='b_names', type=str, nargs='+', default='NO_NAME') 
        for f in args.beams:
            filename = './bodies/' + f
            BEAM_SETTINGS = ''
            with open(filename) as file:
                for line in file.readlines():
                    BEAM_SETTINGS += line.strip() + ' '
            beam_args = beam_parser.parse_known_args(BEAM_SETTINGS.split())[0]
            posx_ind, posy_ind, angle_ind = self.system.make_beam(beam_args.N_sect, beam_args.parent, beam_args.l, beam_args.m, beam_args.Iz, beam_args.fixed, np.array(beam_args.dik), beam_args.b_names)
            self.y0[posx_ind] = beam_args.pos[0]
            self.y0[posy_ind] = beam_args.pos[1]
            self.y0[angle_ind] = beam_args.angle

        self.system.set_forces(args.forces, np.array(args.dis))

    def run(self):
        sol = solve_ivp(self.system.RHS, (self.t0, self.tf), self.y0, method='Radau', max_step=1e-1)
        if self.system.save_forces:
            self.system.force_file.close()

        savedir = './results/' + self.name + '/'

        N = self.system.N_bodies

        q = np.empty((N + 1, len(sol.t)))
        qd = np.empty((N + 1, len(sol.t)))

        q[0, :] = sol.t
        q[1:, :] = sol.y[:N]
        qd[0, :] = sol.t
        qd[1:, :] = sol.y[N:]

        np.savetxt(savedir + 'q.txt', q)
        np.savetxt(savedir + 'qd.txt', qd)

        styles = ['-', '--', '-.']

        fig, ax = plt.subplots(figsize=(13, 8))
        for i in range(N):
            ax.plot(sol.t, sol.y[i], linestyle=styles[i%3], label=f'q[{i + 1}]')
        ax.grid()
        ax.legend(loc='center left')
        fig.savefig('./images/q_explose.pdf', format='pdf')

        fig, ax = plt.subplots(figsize=(13, 8))
        for i in range(N, 2*N):
            ax.plot(sol.t, sol.y[i], linestyle=styles[i%3], label=f'qd[{i - N + 1}]')
        ax.grid()
        ax.legend(loc='center left')
        fig.savefig('./images/qd_explose.pdf', format='pdf')
        plt.show()
