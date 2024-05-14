import sys
import numpy as np
from matplotlib import pyplot as plt

sim_name = sys.argv[1]
data = np.loadtxt('./results/'+sim_name+'/forces.txt', skiprows=1).T

t = data[0]
F = data[1:]

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
ax.plot(t, F[7], label='left')
ax.plot(t, F[9], label='right')
ax.legend()
ax.set_title('Vertical force at the wall')
ax.grid()
fig.savefig('./images/' + sim_name + '_vertical_force.pdf', format='pdf')

fig, ax = plt.subplots(figsize=(13, 8))
ax.plot(t, F[6], label='left')
ax.plot(t, F[8], label='right')
ax.legend()
ax.set_title('Horizontal force at the wall')
ax.grid()
fig.savefig('./images/' + sim_name + '_horizontal_force.pdf', format='pdf')

plt.show()
