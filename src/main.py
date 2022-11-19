import os
import numpy as np
from tqdm import tqdm
from time import time
import argparse

from environment import ENVIRONMENT
from DQN_brain import DQN
from spatial import SPATIAL, track_functions

parser = argparse.ArgumentParser(description='Mobile Hybrid Network MAC Simulator')
# network configurations
parser.add_argument('--n_agents', default=0, type=int, metavar='N', 
					help='number of agents (default 0)', choices=[0, 1],)
parser.add_argument('--n_others', default=2, type=int, metavar='N', 
					help='number of non-agents (default 2)')
parser.add_argument('--packet_length', default=3, type=int, metavar='N', 
					help='packet length (default 3 sub slots)')
parser.add_argument('--guard_length', default=3, type=int, metavar='N', 
					help='guard length (default 3 sub slots)')
parser.add_argument('--sub_slot_length', default=1e7, type=int, metavar='T', 
					help='length of sub time slot (default 1e7 ns)')
parser.add_argument('--delay_max', default=100, type=int, metavar='N', 
					help='maximum delay, or distance upper bound (default 100 sub slots)')
parser.add_argument('--num_sub_slot', default=20, type=int, metavar='N', 
					help='number of sub time slot per time slot (default 20 sub slots')
# inbuilt MAC configurations
parser.add_argument('--frame_length', default=10, type=int, metavar='N', 
					help='length of a TDMA time frame (default 10 time slots)')
parser.add_argument('--tdma_occupancy', default=3, type=int, metavar='N', 
					help='number of TDMA transmissions in a frame (default 3 times)')
parser.add_argument('--aloha_prob', default=0.5, type=float, metavar='q', 
					help='ALOHA transmission probability q (default 0.5)')
parser.add_argument('--window_size', default=2, type=int, metavar='N', 
					help='ALOHA fixed window or exponential backoff window initial window size (default 2 time slots)')
parser.add_argument('--max_backoff', default=2, type=int, metavar='N', 
					help='ALOHA exponential backoff maximum backoff (default 2)')
# simulation modes selection
parser.add_argument('--env_mode', default=1, type=int, metavar='M', 
					help='medium selection, 0 RF, others UAN (default 1)')
parser.add_argument('--env_mac_mode', default=1, type=int, metavar='M', 
					help='inbuilt MAC mode, 0 sync, others async (default 1)')
parser.add_argument('--agent_mac_mode', default=1, type=int, metavar='M', 
					help='agent MAC mode, 0 sync, others async (default 1)')
parser.add_argument('--sink_mode', default=0, type=int, metavar='M', 
					help='agent node type, 0 source node, 1 sink node (default 0)')
# DQN configurations
parser.add_argument('--state_len', default=20, type=int, metavar='M', 
					help='length of super-state (default 20 time slots)')
parser.add_argument('--memory_size', default=1000, type=int, metavar='E', 
					help='size of history state memory (default 1000 super-states)')
parser.add_argument('--replace_target_iter', default=20, type=int, metavar='F', 
					help='update frequency of target network (default 20 time slots)')
parser.add_argument('--train_interval', default=5, type=int, metavar='F', 
					help='training frequency (default 5 time slots)')
parser.add_argument('--batch_size', default=64, type=int, metavar='B', 
					help='training batch size (default 64)')
parser.add_argument('--gamma', default=0.9, type=float, metavar='G', 
					help='discount factor gamma (default 0.9)')
parser.add_argument('--alpha', default=1, type=float, metavar='A', 
					help='Bellman equation learning rate alpha (default 1.0)')
parser.add_argument('--lr', default=0.01, type=float, metavar='LR', 
					help='NN optimizer learning rate (default: 0.01)')
parser.add_argument('--reward_polarity', action='store_true',
					help='if set this flag, the DQN will have a non-zero penalty for the collision')
parser.add_argument('--penalty_factor', default=1, type=float, metavar='P', 
					help='non-zero penalty scale (default 1.0)')
parser.add_argument('--epsilon', default=1, type=float, metavar='e', 
					help='epsilon decay start value (default 1)')
parser.add_argument('--epsilon_min', default=0.01, type=float, metavar='MIN', 
					help='epsilon decay minimum value (default 0.01)')
parser.add_argument('--epsilon_decay', default=0.995, type=float, metavar='D', 
					help='epsilon decay factor (default 0.995)')
# spatial simulator configurations
parser.add_argument('--movable', action='store_true',
					help='if set this flag, the simulator will update delays during runtime')	
parser.add_argument('--time_granularity', default=30e9, type=int, metavar='T', 
					help='time period of spatial simulator (default 30e9 ns)')
parser.add_argument('--distance_init', action='store_true',
					help='if set this flag, the simulator will use the delay to initialize position')
parser.add_argument('--random_init', action='store_true',
					help='if set this flag, the simulator will randomly initialize position')	
# save trace related
parser.add_argument('--save_trace', action='store_true',
					help='if set this flag, the simulator will save throughput')
parser.add_argument('--save_track', action='store_true',
					help='if set this flag, the simulator will save moving track')
parser.add_argument('--max_iter', default=50000, type=int, metavar='N', 
					help='number of iterations (default 50000 time slots)')
parser.add_argument('--log_path', default='../logs/', type=str, metavar='DIR',
					help='log (throughput) path to use (default: ../logs/)')
parser.add_argument('--config_path', default='../configs/', type=str, metavar='DIR',
					help='config (experiment setup) path to use (default: ../configs/)')
parser.add_argument('--track_path', default='../tracks/', type=str, metavar='DIR',
					help='track (moving positions) path to use (default: ../tracks/)')
parser.add_argument('--file_prefix', default='', type=str, metavar='NAME',
					help='file prefix (default: None)')

args = parser.parse_args()

def main(max_iter, env, agent=None, sim_mode=1):
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
			if t > 100 and (t % args.train_interval == 0):
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
# TODO: load saved configurations / parse config files
if __name__ == "__main__":
	n_nodes = args.n_agents + args.n_others # number of nodes
	# nodes_mask = np.random.randint(1, 5, n_nodes)
	# nodes_mask = np.array([2, 4, 2, 4, 1, 4, 3, 3, 2, 2, 2, 3, 4, 2, 1, 2, 4, 1, 3, 4, 1, 3, 2, 4, 2], dtype=int)
	# nodes_mask = np.array([2, 1], dtype=int)
	# nodes_mask = np.array([0, 2], dtype=int)
	# Xuan's case 1 & 2
	# nodes_mask = np.array([0, 1, 2], dtype=int)
	# Xuan's agent-qALOHA coesist
	nodes_mask = 2 * np.ones(n_nodes, dtype=int)
	# the first one should be the agent
	nodes_mask[0] = nodes_mask[0] if args.n_agents == 0 else 0

	# mask used for hybrid network
	# Xuan's case 1
	# delay = np.array([28, 10, 20], dtype=int)
	# Xuan's case 2
	# delay = np.array([13, 13, 13], dtype=int)
	# Xuan's agent-qALOHA coesist
	# delay = np.random.randint(1, 83, n_nodes)
	# Ruoyu's mobility test
	# delay = np.random.randint(1, delay_max, n_nodes)
	delay = np.array([51, 60, 49, 73, 34, 25, 33, 89, 60, 49, 100, 71, 24, 17, 73, 65, 81, 31, 65, 2, 86, 25, 30, 99, 33], dtype=int)
	args.num_sub_slot = 1 if args.env_mode == 0 else 20

	# move frequency in sub time slot
	move_freq = 1 / (args.time_granularity / args.sub_slot_length)

	# unit in meter, can be any positive real number
	# the sptial simulator will randomly generate nodes coordinates as
	# (x, y, z) where x, y, z in [scale * (-0.5, 0.5)]
	scale = 2 * 1500 * (args.delay_max * args.sub_slot_length * 1e-9)
	x = np.array([208.914773, -736.46602, 480.537697, -210.472598, 316.414233,
		 -192.998367, -319.033901, 762.238831, 414.926095, 669.543245,
		 -1421.686803, -565.33843, -144.956256, 84.897915, 1055.82331,
		 -956.418692, 904.669339, 120.11628, -379.682744, 1.969628,
		 -182.404386, -266.67309, -191.007005, -1193.142124, 111.53603], dtype=float)
	y = np.array([702.136641, -271.478263, 524.515919, -905.226902, -398.260894,
		 311.177429, -223.862728, 886.16391, -780.735262, 95.587816,
		 -93.732475, -638.998097, -109.051325, 139.769583, -99.91237,
		 -5.810079, 549.065821, -107.034741, -642.518984, 14.535218,
		 449.60557, 240.730111, -216.290798, 882.4457, -408.671267], dtype=float)
	z = np.array([-220.417229, -408.830472, -184.909633, -579.04278, -37.017475,
	     80.902644, -280.183599, 644.926731, -38.907414, 246.930785,
   		 418.623766, -612.024421, 293.462932, 195.657373, 272.542942,
	     -82.640962, 596.929737, 436.280464, 627.403673, 3.138154,
		 -1195.275479, 107.515006, 325.514385, 54.09675, 256.071297], dtype=float)
	
	# track demo generation
	# scale = 4
	# distance_init = False
	# random_init = True
	# use the helper to generate track functions
	func_helper = track_functions(args.time_granularity)
	track = []
	for i in range(n_nodes):
		# dice = np.random.rand()
		# if dice < 0.33:
		# if i <= 1:
		# 	velocity = func_helper.resultant2component3d(.5)
		# 	func = func_helper.linear(velocity[0], velocity[1], velocity[2])
		# # elif 0.33 <= dice < 0.66:
		# elif 1 < i <= 3:
		# 	velocity = func_helper.resultant2component3d(.5)
		# 	func = func_helper.spiral(np.pi / 30 * (time_granularity * 1e-9), velocity[0] ** 2 + velocity[1] ** 2, velocity[2])
		# else:
		# 	threshold = 3 * time_granularity * 1e-9
		# 	# func = func_helper.backNforth(func_helper.resultant2component3d(0.3), threshold)
		# 	func = func_helper.AUV_assisted(1, 2)
		func = func_helper.AUV_assisted(2, 4)
		track.append(func)

	# saving parameters
	n_iter = int((args.max_iter * args.num_sub_slot + 2 * args.delay_max) * move_freq) + 1
	file_name = f'iter{args.max_iter}_N{n_nodes}_'
	file_timestamp = f'{int(time())}'
	log_suffix = '.txt'
	config_suffix = '.conf'

	print('trace name:')
	print(args.file_prefix + file_name + file_timestamp)

	env = ENVIRONMENT(args.n_agents,
					  args.n_others,
					  nodes_mask[:n_nodes],
					  packet_length = args.packet_length,
				 	  guard_length = args.guard_length,
					  sub_slot_length = args.sub_slot_length,
					  frame_length = args.frame_length,
					  tdma_occupancy = args.tdma_occupancy,
					  aloha_prob = args.aloha_prob,
					  window_size = args.window_size,
					  max_backoff = args.max_backoff,
					  env_mode = args.env_mode,
					  mac_mode = args.env_mac_mode,
					  sink_mode = args.sink_mode,
					  nodes_delay = delay[:n_nodes],
					  num_sub_slot = args.num_sub_slot,
					  movable = args.movable,
					  move_freq = move_freq,
					  save_trace = args.save_trace,
					  n_iter = args.max_iter,
					  log_name = args.log_path + args.file_prefix + file_name + file_timestamp + log_suffix,
					  config_name = args.config_path + args.file_prefix + file_name + file_timestamp + config_suffix,
					  )

	if args.movable:
		spatial = SPATIAL(n_nodes,
						  track,
						  scale = scale,
						  time_granularity = args.time_granularity,
						  distance_init = args.distance_init, distance = env.delay2distance(),
				 		  random_init = args.random_init, x = x[:n_nodes], y = y[:n_nodes], z = z[:n_nodes],
				 		  save_trace = args.save_track, n_iter = n_iter, 
						  file_name = args.track_path + args.file_prefix + file_name + file_timestamp + log_suffix,
						  )
		env.attach_spatial(spatial)

	agent = DQN(args.state_len,
				n_nodes,
				args.num_sub_slot,
				guard_length = args.guard_length,
				memory_size = args.memory_size,
				replace_target_iter = args.replace_target_iter,
				batch_size = args.batch_size,
				learning_rate = args.lr,
				gamma = args.gamma,
				epsilon = args.epsilon,
				epsilon_min = args.epsilon_min,
				epsilon_decay = args.epsilon_decay,
				alpha = args.alpha,
				mac_mode = args.agent_mac_mode,
				sink_mode = args.sink_mode,
				reward_polarity = args.reward_polarity,
				penalty_factor = args.penalty_factor,
				save_trace = args.save_trace,
				config_name = args.config_path + args.file_prefix + file_name + file_timestamp + config_suffix,
				)

	main(args.max_iter, env, agent, args.n_agents)
