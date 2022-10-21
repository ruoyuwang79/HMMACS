import numpy as np
import matplotlib.pyplot as plt

fig_path = '../figs/'
file_path = '../logs/'
file_name = input('Please type in the trace file name (no path, no suffix): ')
fig_suffix = '.png'
file_suffix = '.txt'

for i in file_name.split('_'):
    if 'N' in i:
        n_nodes = int(i[1:])

N = 1000 # moving average window size

Rewards_logs = np.loadtxt(file_path + file_name + file_suffix, dtype=int)
Rewards_logs = np.insert(Rewards_logs, 0, 0, axis=0)
cumsum = np.cumsum(Rewards_logs, axis=0)
Throughput = (cumsum[N:] - cumsum[:-N]) / float(N)

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(n_nodes):
    plt.plot(Throughput[:, i])
    ax.set_ylabel('throughput')
    ax.set_xlabel('time slot')
    print(f'the {i + 1}-th node has throughput {Throughput[-1, i]:.4f}')
plt.savefig(fig_path + file_name + fig_suffix)
print(f'the system throughput is {Throughput[-1].sum():.4f}')
