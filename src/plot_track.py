import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig_path = '../figs/'
file_path = '../tracks/'
file_name = input('Please type in the track file name (no path, no suffix): ')
fig_suffix = '.gif'
file_suffix = '.txt'

for i in file_name.split('_'):
    if 'N' in i:
        n_nodes = int(i[1:])

def animate_scatters(iteration, data, scatters):
	scatters._offsets3d = (data[iteration, 0, :], data[iteration, 1, :], data[iteration, 2, :])
	return scatters

# downsampling rate
downsampling_rate = 10

# load and manipulate data
data = np.loadtxt(file_path + file_name + file_suffix, dtype=float)
data = data.reshape((-1, 3, n_nodes))
data = data[::downsampling_rate, :, :]
data = data[(data != 0).all(2).all(1)]

# randomly generate colors
color = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(n_nodes)]
# for agent (if no, comment)
color[0] = (1, 0, 0)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

scatters = ax.scatter(data[0, 0, :], data[0, 1, :], data[0, 2, :], c=color)

ani = animation.FuncAnimation(fig, animate_scatters, data.shape[0], fargs=(data, scatters), interval=50, blit=False, repeat=True)

for i in range(n_nodes):
	ax.plot(data[:, 0, i], data[:, 1, i], data[:, 2, i], c=color[i])

ani.save(fig_path + file_name + fig_suffix)
