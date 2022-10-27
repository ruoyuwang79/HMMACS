import numpy as np
import matplotlib.pyplot as plt
from spatial import *

test_f = track_functions()
test_s = SPATIAL(3, [test_f.static(), test_f.static(), test_f.linear(3,-4,5)], scale=1)

for i in range(10):
    test_s.step()
    print(test_s[2])
