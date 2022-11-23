import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

exp_path = '../temp/'
log_path = '../logs/'
exp_names = [f for f in listdir(exp_path) if isfile(join(exp_path, f))]
log_names = []
log_suffix = '.txt'

for i in exp_names:
    with open(exp_path + i) as f:
        log_names.append(f.readlines()[1][:-1])
        f.close()

for file_name in log_names:
    for i in file_name.split('_'):
        if 'N' in i:
            n_nodes = int(i[1:])
    print(file_name)
    Rewards_logs = np.loadtxt(log_path + file_name + log_suffix, dtype=int)
    Throughput = Rewards_logs.mean(0)
    print(f'the system throughput is {Throughput.sum():.4f}')
