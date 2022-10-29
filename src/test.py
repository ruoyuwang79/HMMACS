import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from spatial import *

def animate_scatters(iteration, data, scatters):
    scatters._offsets3d = (data[iteration, 0, :], data[iteration, 1, :], data[iteration, 2, :])
    return scatters

n_nodes = 10
test_f = track_functions()
color = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(n_nodes)]
color[0] = (1, 0, 0)
track_funcs = [test_f.linear(color[i][0] - 0.5, color[i][1] - 0.5, color[i][2] - 0.5) if np.random.rand() < 0.5 else test_f.spiral(np.pi / (30 * color[i][0]), 20 * color[i][1], 5 * color[i][2]) for _ in range(n_nodes) for i in range(n_nodes)]
# track_funcs = [test_f.spiral(np.pi / 10, 10, 1) for _ in range(n_nodes)]
track_funcs[0] = test_f.static()
test_s = SPATIAL(n_nodes, track_funcs, scale=1)

time_max = 100
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
data = np.zeros((time_max, 3, n_nodes))
for i in range(time_max):
    test_s.step()
    x, y, z = test_s.get_all_position()
    data[i, 0, :] = x
    data[i, 1, :] = y
    data[i, 2, :] = z
    # ax.scatter(x, y, z)

scatters = ax.scatter(data[0, 0, :], data[0, 1, :], data[0, 2, :], c=color)

ani = animation.FuncAnimation(fig, animate_scatters, time_max, fargs=(data, scatters), interval=100, blit=False, repeat=True)

for i in range(n_nodes):
    ax.plot(data[:, 0, i], data[:, 1, i], data[:, 2, i], c=color[i])

ani.save('../figs/test.gif')
