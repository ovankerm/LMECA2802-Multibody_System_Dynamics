import sys
import numpy as np
from matplotlib import pyplot as plt

sim_name = sys.argv[1]
# ------ FORCES ------
data = np.loadtxt('./results/'+sim_name+'/forces.txt', skiprows=1).T
data_1 = np.loadtxt('./results/sim1/forces.txt', skiprows=1).T

t = data[0]
F = data[1:]

R_Horiz = np.loadtxt('./robotran_data/WallHorF.txt')
R_Vert = np.loadtxt('./robotran_data/WallVertF.txt')

F_norm = np.zeros((len(F)//2, len(F[0])))
for i in range(len(F)//2):
    F_norm[i] = np.power(F[2 * i]*F[2 * i] + F[2 * i + 1]*F[2*i + 1], 0.5)

fig, ax = plt.subplots(figsize=(13, 8))
for i,f in enumerate(F_norm):
    ax.plot(t, f, label=f'f{i+1}')
ax.legend()
ax.grid()
fig.savefig('./images/' + sim_name + '_forces.pdf', format='pdf')

fig, ax = plt.subplots(figsize=(13, 8))
ax.plot(F[18], F[7], 'r-', label='Second Simulation, left side')
ax.plot(F[18], F[9], 'r--', label='Second Simulation, right side')
ax.plot(50e3 * data_1[0]/10, data_1[8], 'g-', label='First Simulation, left side')
ax.plot(50e3 * data_1[0]/10, data_1[10], 'g--', label='First Simulation, right side')
ax.legend()
ax.set_title('Vertical force on the wall')
ax.set_ylabel('Vertical force [N]')
ax.set_xlabel('Wind force [N]')
ax.grid()
fig.savefig('./images/' + sim_name + '_vertical_force.pdf', format='pdf')

fig, ax = plt.subplots(figsize=(13, 8))
ax.plot(F[18], F[6], 'r-', label='Second Simulation, left side')
ax.plot(F[18], F[8], 'r--', label='Second Simulation, right side')
ax.plot(50e3 * data_1[0]/10, data_1[7], 'g-', label='First Simulation, left side')
ax.plot(50e3 * data_1[0]/10, data_1[9], 'g--', label='First Simulation, right side')
ax.legend()
ax.set_title('Horizontal force on the wall')
ax.set_ylabel('Horizontal force [N]')
ax.set_xlabel('Wind force [N]')
ax.grid()
fig.savefig('./images/' + sim_name + '_horizontal_force.pdf', format='pdf')


# ------ POSITIONS ------
data = np.loadtxt('./results/'+sim_name+'/positions.txt', skiprows=1).T
data_1 = np.loadtxt('./results/'+'sim1'+'/positions.txt', skiprows=1).T
data_rob = np.loadtxt('./robotran_data/WallHorP.txt').T

t = data[0]
pos = data[1:]

fig, ax = plt.subplots(figsize=(13, 8))
ax.plot(50e3 * t/50, pos[0], 'k-', label='Second Simulation')
ax.plot(50e3 * data_1[0]/10, data_1[1], 'g-', label='First Simulation')
ax.set_ylabel('Horizontal displacement [m]')
ax.set_xlabel('Wind force [N]')
ax.set_title('Horizontal displacement of the top point (zoomed in)')
ax.set_xlim(40e3, 50e3)
ax.set_ylim(0.12, 0.16)
ax.legend()
ax.grid()
fig.savefig('./images/' + sim_name + '_horiz_displacement.pdf', format='pdf')


# ------ Q and QD ------
data = np.loadtxt('./results/'+sim_name+'/q.txt')

t = data[0]
q = data[1:]

data = np.loadtxt('./results/'+sim_name+'/qd.txt')

t = data[0]
qd = data[1:]
N = len(qd)

styles = ['-', '--', '-.']

fig, ax = plt.subplots(figsize=(13, 8))
for i in range(N):
    ax.plot(t, q[i], linestyle=styles[i%3], label=f'q[{i + 1}]')
ax.grid()
ax.legend(loc='center left')
fig.savefig(f'./images/q_{sim_name}.pdf', format='pdf')

fig, ax = plt.subplots(figsize=(13, 8))
for i in range(N):
    ax.plot(t, qd[i], linestyle=styles[i%3], label=f'qd[{i + 1}]')
ax.grid()
ax.legend(loc='center left')
fig.savefig(f'./images/qd_{sim_name}.pdf', format='pdf')
plt.show()
