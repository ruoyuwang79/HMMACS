import numpy as np

# Current & AUV movement velocity can be accumulated, 
# the simulator can support a series of functions to a single node
# then the system can have different considerations simultaneously
class func_matrix():
	def __init__(self, func_list):
		self.func_list = func_list
	
	def __call__(self, v, a):
		for func in self.func_list:
			v, a = func(v, a)
		return v, a

# to maximize the efficiency, once initlized, cannot add new nodes
# just simulate given configuration
# different scenarios should be designed outside
# e.g. surface sensors: initlize z = np.zeros, track has no vz
class SPATIAL():
	def __init__(self, n_nodes: int, track: list, 
				 scale: int = 2e3, time_granularity: int = 1e7, 
				 distance_init: bool = False, distance: np.array = None,
				 random_init: bool = True, x: np.array = None, y: np.array = None, z: np.array = None,
				 save_trace: bool = False, n_iter: int = 1, file_name: str = ''):
		# number of source nodes
		self.n_nodes = n_nodes
		# all nodes track function
		# track function input is current velocity and acceleration
		self.track = track
		# largest coordinate (in meter) to origin (sink)
		self.scale = scale
		# in nano seconds (1e-9 s)
		self.time_granularity = time_granularity
		self.distance_init = distance_init
		self.random_init = random_init
		if self.distance_init:
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
		elif self.random_init:
			# randomly generate a cube
			self.x = self.scale * (np.random.rand(self.n_nodes) - 0.5)
			self.y = self.scale * (np.random.rand(self.n_nodes) - 0.5)
			self.z = self.scale * (np.random.rand(self.n_nodes) - 0.5)
		else:
			# use the given coordinate
			self.x = x
			self.y = y
			self.z = z
		
		# initialize all nodes velocity static
		self.vx = np.zeros(self.n_nodes, dtype=float)
		self.vy = np.zeros(self.n_nodes, dtype=float)
		self.vz = np.zeros(self.n_nodes, dtype=float)
		# initialize all nodes acceleration static
		self.ax = np.zeros(self.n_nodes, dtype=float)
		self.ay = np.zeros(self.n_nodes, dtype=float)
		self.az = np.zeros(self.n_nodes, dtype=float)

		self.save_trace = save_trace
		if self.save_trace:
			self.n_iter = n_iter
			self.file_name = file_name
			self.step_counter = 0
			self.history_positions = np.zeros((self.n_iter + 1, 3, self.n_nodes))
			self.history_positions[self.step_counter, 0, :] = self.x
			self.history_positions[self.step_counter, 1, :] = self.y
			self.history_positions[self.step_counter, 2, :] = self.z
			self.step_counter += 1

	def __getitem__(self, idx):
		return (self.x[idx], self.y[idx], self.z[idx])

	def get_all_position(self):
		return (self.x, self.y, self.z)

	def get_distance(self):
		return np.sqrt(self.x**2 + self.y**2 + self.z**2)

	# x = vt, v = v + a
	# return the new speed
	def update_a(self):
		dvda = np.array([self.track[i]((self.vx[i], self.vy[i], self.vz[i]), (self.ax[i], self.ay[i], self.az[i])) for i in range(self.n_nodes)], dtype=float)
		dv = dvda[:, 0, :]
		da = dvda[:, 1, :]
		self.ax = da[:, 0]
		self.ay = da[:, 1]
		self.az = da[:, 2]
		return (dv[:, 0], dv[:, 1], dv[:, 2])

	# use the new speed to replace the original
	def update_v(self, dvx, dvy, dvz):
		self.vx = dvx
		self.vy = dvy
		self.vz = dvz

	# velocity in m/s, time granularity in ns, distance in m
	def update_x(self):
		self.x += self.vx * self.time_granularity * 1e-9
		self.y += self.vy * self.time_granularity * 1e-9
		self.z += self.vz * self.time_granularity * 1e-9

	# API for environment (after attach, env will call this)
	def step(self):
		dvx, dvy, dvz = self.update_a()
		self.update_v(dvx, dvy, dvz)
		self.update_x()
		if self.save_trace:
			self.history_positions[self.step_counter, 0, :] = self.x
			self.history_positions[self.step_counter, 1, :] = self.y
			self.history_positions[self.step_counter, 2, :] = self.z
			self.step_counter += 1

	# API for environment (after attach, env will call this at the end)
	def finalize(self):
		if self.save_trace:
			# cut off all zero tails and store trace
			np.savetxt(self.file_name, self.history_positions[:self.step_counter, :, :].reshape((-1, 3 * self.n_nodes)), fmt='%f')

class track_functions():
	# all velocity in pre-defined functions are per step
	# use the converter to translate before generate
	def __init__(self, time_granularity: int = 1e7,):
		super(track_functions, self).__init__()
		self.time_granularity = time_granularity

	# randomly decompose a resultant velocity
	# functionality: uniformly pick a random direction in 3d
	def resultant2component3d(self, resultant):
		dvx = np.random.randn()
		dvy = np.random.randn()
		dvz = np.random.randn()
		factor = resultant / np.sqrt(dvx ** 2 + dvy ** 2 + dvz ** 2)
		return (factor * dvx, factor * dvy, factor * dvz)

	# randomly decompose a resultant velocity
	# functionality: uniformly pick a random direction in 2d
	def resultant2component2d(self, resultant):
		theta = 2 * np.pi * np.random.rand()
		return (resultant * np.cos(theta), resultant * np.sin(theta), 0)

	# from step velocity (m / step) to normalized (m / s)
	def step2norm(self, velocity):
		vx_norm = velocity[0] / (self.time_granularity * 1e-9)
		vy_norm = velocity[1] / (self.time_granularity * 1e-9)
		vz_norm = velocity[1] / (self.time_granularity * 1e-9)
		return (vx_norm, vy_norm, vz_norm)

	# from normalized (m / s) to step velocity (m / step)
	def norm2step(self, velocity_norm):
		vx = velocity_norm[0] * (self.time_granularity * 1e-9)
		vy = velocity_norm[1] * (self.time_granularity * 1e-9)
		vz = velocity_norm[2] * (self.time_granularity * 1e-9)
		return (vx, vy, vz)

	def static(self):
		return lambda v, a: ((0, 0, 0), (0, 0, 0))

	def linear(self, dvx, dvy, dvz):
		return lambda v, a: ((dvx, dvy, dvz), (0, 0, 0))

	def spiral(self, angular_velocity, velocity, dvz):
		def spiral_func(v, a):
			epsilon = 1e-7
			new_theta = np.arctan(v[0] / (v[1] + epsilon)) + angular_velocity + (np.pi if v[1] < 0 else 0)
			dvx = np.sqrt(velocity) * np.cos(new_theta)
			dvy = np.sqrt(velocity) * np.sin(new_theta)
			return ((dvx, dvy, dvz), (0, 0, 0))
		return spiral_func
	
	def backNforth(self, acceleration, threshold):
		def back_and_forth(v, a):
			ax = a[0] if a[0] != 0 else acceleration[0]
			ay = a[1] if a[1] != 0 else acceleration[1]
			az = a[2] if a[2] != 0 else acceleration[2]
			if (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) >= threshold ** 2:
				ax *= -1
				ay *= -1
				az *= -1
			dvx = v[0] + ax
			dvy = v[1] + ay
			dvz = v[2] + az
			return ((dvx, dvy, dvz), (ax, ay, az))
		return back_and_forth

	### used to reproduce, developing ###
	# UW-ALOHA-QM case A
	def moored(self, min_v, max_v):
		return lambda v, a: (self.resultant2component2d(np.random.uniform(min_v, max_v)), (0, 0, 0))

	def floating(self):
		return lambda v, a: ((0, 0, 0), (0, 0, 0))

	def AUV_assisted(self, min_v, max_v):
		return lambda v, a: (self.resultant2component3d(np.random.uniform(min_v, max_v)), (0, 0, 0))

	def AUV_network(self, velocity):
		return lambda v, a: (self.resultant2component3d(velocity), (0, 0, 0))


	# This idea is very tricky for implementation
	# TODO: time sequence function, in a period, first partition of time takes f1
	# second partition of time takes f2, time partition can be configured
