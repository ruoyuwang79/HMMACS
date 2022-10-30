import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# to maximize the efficiency, once initlized, cannot add new nodes
class SPATIAL():
	def __init__(self, n_nodes, track, 
				 scale=5e3, time_granularity=1e7, 
				 random_init=True, x=None, y=None, z=None,
				 save_track=True, color=None, n_iter=1, fig_name=''):
		# number of source nodes
		self.n_nodes = n_nodes
		# all nodes track function
		# function is direction of acceleration based on position
		self.track = track
		# largest coordinate (in meter) to origin (sink)
		self.scale = scale
		# in nano seconds
		self.time_granularity = time_granularity
		self.random_init = random_init
		if self.random_init:
			self.x = self.scale * (2 * np.random.rand(self.n_nodes) - 1)
			self.y = self.scale * (2 * np.random.rand(self.n_nodes) - 1)
			self.z = self.scale * (2 * np.random.rand(self.n_nodes) - 1)
		else:
			self.x = x
			self.y = y
			self.z = z
		self.vx = np.zeros(self.n_nodes, dtype=float)
		self.vy = np.zeros(self.n_nodes, dtype=float)
		self.vz = np.zeros(self.n_nodes, dtype=float)

		# for drawer
		self.save_track = save_track
		if self.save_track:
			self.color = color
			self.n_iter = n_iter
			self.step_counter = 0
			self.history_positions = np.zeros((self.n_iter + 1, 3, self.n_nodes))
			self.history_positions[self.step_counter, 0, :] = self.x
			self.history_positions[self.step_counter, 1, :] = self.y
			self.history_positions[self.step_counter, 2, :] = self.z
			self.fig_name = fig_name

	def __getitem__(self, idx):
		return (self.x[idx], self.y[idx], self.z[idx])

	def get_all_position(self):
		return self.x, self.y, self.z

	# TODO: add the capability to get distance of any given nodes pair
	def get_distance(self):
		return np.sqrt(self.x**2 + self.y**2 + self.z**2)

	# return the normalize new speed
	def get_direction(self):
		dv = np.array([self.track[i](self.vx[i], self.vy[i], self.vz[i]) for i in range(self.n_nodes)], dtype=float)
		dvx = dv[:, 0]
		dvy = dv[:, 1]
		dvz = dv[:, 2]
		factor = np.sqrt(dvx**2 + dvy**2 + dvz**2)
		dvx[factor != 0] /= factor[factor != 0]
		dvy[factor != 0] /= factor[factor != 0]
		dvz[factor != 0] /= factor[factor != 0]
		return dvx, dvy, dvz

	# use the new speed to replace the original
	def update_v(self, dvx, dvy, dvz):
		# self.vx += dvx
		# self.vy += dvy
		# self.vz += dvz
		# self.vx /= 2
		# self.vy /= 2
		# self.vz /= 2
		self.vx = dvx
		self.vy = dvy
		self.vz = dvz

	# velocity in m/s, time granularity in ns, distance in m
	def update_position(self):
		self.x += self.vx * self.time_granularity * 1e-9
		self.y += self.vy * self.time_granularity * 1e-9
		self.z += self.vz * self.time_granularity * 1e-9

	def step(self):
		dvx, dvy, dvz = self.get_direction()
		self.update_v(dvx, dvy, dvz)
		self.update_position()
		self.step_counter += 1
		self.history_positions[self.step_counter, 0, :] = self.x
		self.history_positions[self.step_counter, 1, :] = self.y
		self.history_positions[self.step_counter, 2, :] = self.z

	def get_color(self):
		return self.color

	# used to visualize
	def plot_track(self):
		assert self.save_track
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')

		def animate_scatters(iteration, data, scatters):
			scatters._offsets3d = (data[iteration, 0, :], data[iteration, 1, :], data[iteration, 2, :])
			return scatters

		scatters = ax.scatter(self.history_positions[0, 0, :], self.history_positions[0, 1, :], self.history_positions[0, 2, :], c=self.color)

		ani = animation.FuncAnimation(fig, animate_scatters, self.n_iter, fargs=(self.history_positions, scatters), interval=100, blit=False, repeat=True)

		for i in range(self.n_nodes):
			ax.plot(self.history_positions[:, 0, i], self.history_positions[:, 1, i], self.history_positions[:, 2, i], c=self.color[i])

		ani.save(self.fig_name)

class track_functions():
	def __init__(self):
		print('track function helper initlized')
	
	def static(self):
		return lambda vx, vy, vz: (0, 0, 0)

	def linear(self, dx, dy, dz):
		return lambda vx, vy, vz: (dx, dy, dz)

	def spiral(self, angular_velocity, velocity, dz):
		def spiral_func(vx, vy, vz):
			epsilon = 1e-7
			new_theta = np.arctan(vx / (vy + epsilon)) + angular_velocity + (np.pi if vy < 0 else 0)
			dvx = np.sqrt(velocity) * np.sin(new_theta)
			dvy = np.sqrt(velocity) * np.cos(new_theta)
			dvz = dz
			return dvx, dvy, dvz
		return spiral_func
