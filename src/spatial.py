import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# to maximize the efficiency, once initlized, cannot add new nodes
class SPATIAL():
	def __init__(self, n_nodes, track, 
				 scale=2e3, time_granularity=1e7, 
				 random_init=True, distance=None,
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
			self.x = self.scale * (np.random.rand(self.n_nodes) - 0.5)
			self.y = self.scale * (np.random.rand(self.n_nodes) - 0.5)
			self.z = self.scale * (np.random.rand(self.n_nodes) - 0.5)
		else:
			# give distance, randomly pick coordination on the smphere
			# algorithm based on https://mathworld.wolfram.com/SpherePointPicking.html
			# algorithm created by Muller 1959 and Marsaglia 1972
			x = np.random.randn(self.n_nodes)
			y = np.random.randn(self.n_nodes)
			z = np.random.randn(self.n_nodes)
			factor = distance / np.sqrt(x**2 + y**2 + z**2)
			self.x = factor * x
			self.y = factor * y
			self.z = factor * z
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

	# return the new speed
	def get_v(self):
		dv = np.array([self.track[i](self.vx[i], self.vy[i], self.vz[i]) for i in range(self.n_nodes)], dtype=float)
		return dv[:, 0], dv[:, 1], dv[:, 2]

	# use the new speed to replace the original
	def update_v(self, dvx, dvy, dvz):
		self.vx = dvx
		self.vy = dvy
		self.vz = dvz

	# velocity in m/s, time granularity in ns, distance in m
	def update_position(self):
		self.x += self.vx * self.time_granularity * 1e-9
		self.y += self.vy * self.time_granularity * 1e-9
		self.z += self.vz * self.time_granularity * 1e-9

	def step(self):
		dvx, dvy, dvz = self.get_v()
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

		# ani = animation.FuncAnimation(fig, animate_scatters, int(self.step_counter / 1000), fargs=(self.history_positions[::1000, :, :], scatters), interval=50, blit=False, repeat=True)
		ani = animation.FuncAnimation(fig, animate_scatters, int(self.step_counter), fargs=(self.history_positions, scatters), interval=50, blit=False, repeat=True)

		for i in range(self.n_nodes):
			ax.plot(self.history_positions[:self.step_counter, 0, i], self.history_positions[:self.step_counter, 1, i], self.history_positions[:self.step_counter, 2, i], c=self.color[i])

		ani.save(self.fig_name)

class track_functions():
	def __init__(self):
		super(track_functions, self).__init__()

	def static(self):
		return lambda vx, vy, vz: (0, 0, 0)

	def linear(self, dvx, dvy, dvz):
		return lambda vx, vy, vz: (dvx, dvy, dvz)

	def spiral(self, angular_velocity, velocity, dvz):
		def spiral_func(vx, vy, vz):
			epsilon = 1e-7
			new_theta = np.arctan(vx / (vy + epsilon)) + angular_velocity + (np.pi if vy < 0 else 0)
			dvx = np.sqrt(velocity) * np.sin(new_theta)
			dvy = np.sqrt(velocity) * np.cos(new_theta)
			return dvx, dvy, dvz
		return spiral_func
