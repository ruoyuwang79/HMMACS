import numpy as np
from spatial import *

n_nodes = 10
time_max = 100
test_f = track_functions()
color = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(n_nodes)]
color[0] = (1, 0, 0)
track_funcs = [test_f.linear(color[i][0] - 0.5, color[i][1] - 0.5, color[i][2] - 0.5) if np.random.rand() < 0.5 else test_f.spiral(np.pi / (30 * color[i][0]), 20 * color[i][1], 5 * color[i][2]) for _ in range(n_nodes)]
track_funcs[0] = test_f.static()
test_s = SPATIAL(n_nodes, track_funcs, scale=1, save_track=True, color=color, n_iter=time_max, fig_name='../figs/test.gif')

for i in range(time_max):
	test_s.step()
test_s.plot_track()
