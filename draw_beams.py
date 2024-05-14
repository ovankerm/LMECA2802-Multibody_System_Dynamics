from matplotlib import pyplot as plt
import numpy as np
import sys

sim_name = sys.argv[1]
data = np.loadtxt('./results/' + sim_name + '/q.txt')

indices = [1, 8, 15]
times = [0, 6 * len(data[0])//7, -1]
c = ['k-', 'b-', 'r-']
R = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
fig, ax = plt.subplots()

for k, t in enumerate(times):
    q = data[:, t]
    for i in indices:
        p0 = q[i:i+2]
        a0 = q[i+2]
        points = np.zeros((6, 2))

        points[2] = p0 + R(a0) @ np.array([-1, 0])
        points[1] = points[2] + R(q[i+3] + a0) @ np.array([-2, 0])
        points[0] = points[1] + R(q[i+4] + q[i+3] + a0) @ np.array([-2, 0])
        points[3] = p0 + R(a0) @ np.array([1, 0])
        points[4] = points[3] + R(q[i+5] + a0) @ np.array([2, 0])
        points[5] = points[4] + R(q[i+6] + q[i+5] + a0) @ np.array([2, 0])

        ax.plot(points[:, 0], points[:, 1], c[k], label=f't = {q[0]}')

ax.legend()
ax.grid()
plt.show()
