import numpy as np

# to maximize the efficiency, once initlized, cannot add new nodes
class SPATIAL():
    def __init__(self, n_nodes, scale=1, time_granularity=1e7, random_init=True, x=None, y=None, z=None):
        # number of source nodes
        self.n_nodes = n_nodes
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

    # TODO: add the capability to get distance of any given nodes pair
    def distance(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
