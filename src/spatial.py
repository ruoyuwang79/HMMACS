import numpy as np
import matplotlib.pyplot as plt

# to maximize the efficiency, once initlized, cannot add new nodes
class SPATIAL():
    def __init__(self, n_nodes, track, scale=5e3, time_granularity=1e7, random_init=True, x=None, y=None, z=None):
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

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx], self.z[idx])

    def get_all_position(self):
        return self.x, self.y, self.z

    # TODO: add the capability to get distance of any given nodes pair
    def get_distance(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    # return the normalize new speed
    def get_direction(self):
        dv = np.array([self.track[i](self.x[i], self.y[i], self.z[i]) for i in range(self.n_nodes)], dtype=float)
        dvx = dv[:, 0]
        dvy = dv[:, 1]
        dvz = dv[:, 2]
        factor = np.sqrt(dvx**2 + dvy**2 + dvz**2)
        dvx[factor != 0] /= factor[factor != 0]
        dvy[factor != 0] /= factor[factor != 0]
        dvz[factor != 0] /= factor[factor != 0]
        return dvx, dvy, dvz

    # current scheme: given new speed, use the mean of them to update
    # theoretical base: slope calculation
    def update_v(self, dvx, dvy, dvz):
        self.vx += dvx
        self.vy += dvy
        self.vz += dvz
        self.vx /= 2
        self.vy /= 2
        self.vz /= 2

    # velocity in m/s, time granularity in ns, distance in m
    def update_position(self):
        self.x += self.vx * self.time_granularity * 1e-9
        self.y += self.vy * self.time_granularity * 1e-9
        self.z += self.vz * self.time_granularity * 1e-9

    def step(self):
        dvx, dvy, dvz = self.get_direction()
        self.update_v(dvx, dvy, dvz)
        self.update_position()

    # used to visualize
    def drawer(self):
        pass

class track_functions():
    def __init__(self):
        print('track function helper initlized')
    
    def static(self):
        return lambda x, y, z: (0, 0, 0)

    def linear(self, dx, dy, dz):
        return lambda x, y, z: (dx, dy, dz)
