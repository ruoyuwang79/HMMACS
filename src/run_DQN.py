import os
import numpy as np
from tqdm import tqdm
from time import time

from environment import ENVIRONMENT
from DQN_brain import DQN

def main(max_iter, env, agent=None, sim_mode=1):
	# other hyper parameters
	train_interval = 5

	print('------------------------------------------')
	print('---------- Start processing ... ----------')
	print('------------------------------------------')

	# kickoff
	state = agent.kickoff() if sim_mode == 1 else None
	# simulate in time slot axis
	for t in tqdm(range(max_iter)): 
		if sim_mode == 1:
			action = agent.choose_action(state)
			obs = env.step(action)
			next_state = agent.step(action, obs, state)
			if t > 100 and (t % train_interval == 0):
				agent.learn()
			state = next_state
		else:
			env.step()

	print('total transmitted packet size')
	print(env.finalize())
	if sim_mode == 1:
		agent.finalize()

	print('------------------------------------------')
	print('---------- Finished, log stored ----------')
	print('------------------------------------------')
		
# export TF_CPP_MIN_LOG_LEVEL=3 to disable all verbose
# Level | Level for Humans | Level Description                  
# ------|------------------|------------------------------------ 
# 0     | DEBUG            | [Default] Print all messages       
# 1     | INFO             | Filter out INFO messages           
# 2     | WARNING          | Filter out INFO & WARNING messages 
# 3     | ERROR            | Filter out all messages   

# set running configurations here
# TODO: load saved configurations
if __name__ == "__main__":
	n_agents = 1
	n_others = 9
	n_nodes = n_agents + n_others # number of nodes
	nodes_mask = np.random.randint(1, 5, n_nodes)
	# Xuan's case 1 & 2
	# nodes_mask = np.array([0, 1, 2], dtype=int)
	# Xuan's agent-qALOHA coesist
	# nodes_mask = 2 * np.ones(n_nodes, dtype=int)
	# the first one should be the agent
	nodes_mask[0] = nodes_mask[0] if n_agents == 0 else 0
	
	packet_length = 3
	guard_length = 3
	# 1e7 * 1e-9 s
	sub_slot_length = 1e7

	tdma_occupancy = 3
	tdma_period = 10
	aloha_prob = 0.2
	window_size = 2
	max_backoff = 2

	env_mode = 1
	env_mac_mode = 1
	agent_mac_mode = 1
	sink_mode = 0
	reward_polarity = False

	# mask used for hybrid network
	# Xuan's case 1
	# delay = np.array([28, 10, 20], dtype=int)
	# Xuan's case 2
	# delay = np.array([13, 13, 13], dtype=int)
	# Xuan's agent-qALOHA coesist
	# delay = np.random.randint(1, 83, n_nodes)
	# Ruoyu's mobility test (will be overwritten by the sptial)
	delay = np.random.randint(1, 133, n_nodes)
	num_sub_slot = 20

	state_len = 20 # state length (in # of time slots)
	memory_size = 1000 # memory size (in # of states)
	replace_target_iter = 20 # target network update frequency (in # of time slots)
	batch_size = 64 # mini-batch size
	gamma = 0.9 # discount factor
	alpha = 1 # Bellman equation learning rate
	learning_rate = 0.01 # NN optimizer learning rate
	penalty_factor = 1

	epsilon = 1
	epsilon_min = 0.01
	epsilon_decay = 0.995
	
	movable = True
	# unit in meter, can be any positive real number
	# the sptial simulator will randomly generate nodes coordinates as
	# (x, y, z) where x, y, z in [mobility * (0, 1)]
	mobility = 2000
	# move frequency in sub time slot
	move_freq = 1

	save_trace = True
	max_iter = 10000
	log_path = '../logs/'
	config_path = '../configs/'
	fig_path = '../figs/'
	file_prefix = 'demo_'
	file_name = f'iter{max_iter}_N{n_nodes}_'
	file_timestamp = f'{int(time())}'
	log_suffix = '.txt'
	config_suffix = '.conf'
	fig_suffix = '.gif'

	print('trace name:')
	print(file_prefix + file_name + file_timestamp)

	env = ENVIRONMENT(n_agents,
					  n_others,
					  nodes_mask,
					  packet_length=packet_length,
				 	  guard_length=guard_length,
					  sub_slot_length=sub_slot_length,
					  tdma_occupancy = tdma_occupancy,
					  tdma_period = tdma_period,
					  aloha_prob = aloha_prob,
					  window_size = window_size,
					  max_backoff = max_backoff,
					  env_mode = env_mode,
					  mac_mode = env_mac_mode,
					  sink_mode = sink_mode,
					  nodes_delay = delay,
					  num_sub_slot = num_sub_slot,
					  movable = movable,
					  mobility = mobility,
					  move_freq = move_freq,
					  save_trace = save_trace,
					  n_iter = max_iter,
					  log_name = log_path + file_prefix + file_name + file_timestamp + log_suffix,
					  config_name = config_path + file_prefix + file_name + file_timestamp + config_suffix,
				 	  fig_name = fig_path + file_prefix + file_name + file_timestamp + fig_suffix,
					 )

	agent = DQN(state_len,
				n_nodes,
				num_sub_slot,
				guard_length = guard_length,
				memory_size = memory_size,
				replace_target_iter = replace_target_iter,
				batch_size = batch_size,
				learning_rate = learning_rate,
				gamma = gamma,
				epsilon = epsilon,
				epsilon_min = epsilon_min,
				epsilon_decay = epsilon_decay,
				alpha = alpha,
				mac_mode = agent_mac_mode,
				sink_mode = sink_mode,
				reward_polarity = reward_polarity,
				penalty_factor = penalty_factor,
				save_trace = save_trace,
				config_name = config_path + file_prefix + file_name + file_timestamp + config_suffix,
			   )

	main(max_iter, env, agent, n_agents)
