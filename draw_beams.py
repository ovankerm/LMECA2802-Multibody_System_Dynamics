from matplotlib import pyplot as plt
import numpy as np
import sys

sim_name = sys.argv[1]
data = np.loadtxt('./results/' + sim_name + '/q.txt')
t_sim = data[0]
q_sim = data[1:]
N_beams = len(q_sim)//7
half_t = .5 * (t_sim[-1] + t_sim[0])
first_ind = np.where(t_sim >= half_t)[0][0]

times = [0, first_ind, -1]
lengths = [2, 2, 2, 1, 0.8660254]
c = ['k-', 'b-', 'r-']
R = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

fig, ax = plt.subplots(figsize=(10, 8))
for k, t in enumerate(times):
    q = q_sim[:, t]
    for j in range(N_beams):
        l = lengths[j]
        i = 7 * j
        p0 = q[i:i+2]
        a0 = q[i+2]
        points = np.zeros((6, 2))

        points[2] = p0 + R(a0) @ np.array([-l/2, 0])
        points[1] = points[2] + R(q[i+3] + a0) @ np.array([-l, 0])
        points[0] = points[1] + R(q[i+4] + q[i+3] + a0) @ np.array([-l, 0])
        points[3] = p0 + R(a0) @ np.array([l/2, 0])
        points[4] = points[3] + R(q[i+5] + a0) @ np.array([l, 0])
        points[5] = points[4] + R(q[i+6] + q[i+5] + a0) @ np.array([l, 0])

        if j == 0:
            ax.plot(points[:, 0], points[:, 1], c[k], label='t = %.2f'%(t_sim[t]))
        else:
            ax.plot(points[:, 0], points[:, 1], c[k])

ax.legend()
ax.grid()
ax.set_aspect('equal')
fig.savefig('./images/beam_'+sim_name+'.pdf', format='pdf')
plt.show()
