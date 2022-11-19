import numpy as np

file_path = '../tracks/'
file_name = input('Please type in the track file name (no path, no suffix): ')
file_suffix = '.txt'

for i in file_name.split('_'):
    if 'N' in i:
        n_nodes = int(i[1:])

# load and manipulate data
data = np.loadtxt(file_path + file_name + file_suffix, dtype=float)
data = data.reshape((-1, 3, n_nodes))

idx = int(input(f'Please type in the time index you want to query (max-{data.shape[0]}): '))

# get xyz
print('x:')
print(data[idx, 0, :])
print('y:')
print(data[idx, 1, :])
print('z:')
print(data[idx, 2, :])

# get delay
# distance = np.sqrt(data[idx, 0, :] ** 2 + data[idx, 1, :] ** 2 + data[idx, 2, :] ** 2)
# sub_slot_length = 1e7
# delay = -(((distance / 1500) * 1e9) // -sub_slot_length).astype(int)
# print(delay)

