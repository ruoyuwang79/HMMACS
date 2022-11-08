import numpy as np 

# Optimized for performance
# Note: not multi-thread safe
class DELAY_QUEUE(object):
	def __init__(self, delay: int, reserve: int = 2):
		super(DELAY_QUEUE, self).__init__()
		self.delay = delay if delay > 0 else 1
		self.n_transmission = 0
		# use the reserve trick to accelerate execution
		self.delay_channel = np.zeros(reserve, dtype=int)
		self.packet_length = np.zeros(reserve, dtype=int)
		self.packet_data = -np.ones(reserve, dtype=int)
	
	# Note: the model assumes that packets have no overlapping
	# if the previous transmitted packet longer than the gap between two transmissions
	# un-expectable behavior may happen
	# time spent is maintained by countdown
	def push(self, length: int, packet_data: int):
		# lack of space, enlarge
		if self.delay_channel.size <= self.n_transmission + 1:
			# double the size each time
			new_delay_channel = np.zeros(2 * self.delay_channel.size, dtype=int)
			new_packet_length = np.zeros(2 * self.packet_length.size, dtype=int)
			new_packet_data = -np.ones(2 * self.packet_data.size, dtype=int)
			# copy and push a new item
			new_delay_channel[1:self.delay_channel.size + 1] = self.delay_channel
			new_packet_length[1:self.packet_length.size + 1] = self.packet_length
			new_packet_data[1:self.packet_data.size + 1] = self.packet_data
			new_delay_channel[0] = self.delay
			new_packet_length[0] = length
			new_packet_data[0] = packet_data
			
			self.delay_channel = new_delay_channel
			self.packet_length = new_packet_length
			self.packet_data = new_packet_data
		else:
			# push an transmission to the queue
			self.delay_channel[1:self.n_transmission + 1] = self.delay_channel[:self.n_transmission]
			self.packet_length[1:self.n_transmission + 1] = self.packet_length[:self.n_transmission]
			self.packet_data[1:self.n_transmission + 1] = self.packet_data[:self.n_transmission]
			self.delay_channel[0] = self.delay
			self.packet_length[0] = length
			self.packet_data[0] = packet_data
		self.n_transmission += 1

	def top(self):
		if self.n_transmission > 0:
			return self.delay_channel[self.n_transmission - 1], self.packet_length[self.n_transmission - 1], self.packet_data[self.n_transmission - 1]
		else:
			return 0xffffffff, -1, -1

	def pop(self):
		# check the oldest packet in the queue
		time_remain, top_packet_len, packet_data = self.top()
		# if arrive, return 1
		if time_remain <= 0:
			# the last time slot of transmission, erase
			if time_remain + top_packet_len <= 1:
				# lazy erase
				self.n_transmission -= 1
				# only the finished can return data
				return -1, packet_data
			return 1, -1
		# if nothing arrive or empty queue, return 0
		return 0, -1
	
	# API for mobility update
	def update_delay(self, new_delay: int):
		self.delay = new_delay if new_delay > 0 else 1

	# API between CHANNEL and DELAY_QUEUE
	# Input: length-the transmission packet length at current sub time slot
	# 		 if no transmission, set length = 0
	# Output: if packet arrive at this time slot, return 1, 
	#         if it is the finishing of the packet, return -1, otherwise 0
	# Note: use the cumulative absolute value of the return to get packet size
	#       use the -1 to indicate finish
	# Note: becuase the step behavior is push first, then time slot + 1, then pop
	#       therefore, the minimum of delay is 1, 
	#       if 0 was set, cannot distinguish the difference between packet length 1 and 2
	# Note: the model can be treated as pushing at the begining edge of the time slot
	#       poping at the finishing edge of the time slot
	def step(self, length: int = 1, packet_data: int = -1):
		# only packet length larger than 0 means a transmit
		if length > 0:
			self.push(length, packet_data)
		self.delay_channel -= 1
		return self.pop()

# collection of all-nodes full-duplex channels
class CHANNEL(object):
	def __init__(self, n_nodes: int, delay: np.array):
		super(CHANNEL, self).__init__()
		# number of nodes of the system
		self.n_nodes = n_nodes
		# delay is an integer vector with n_nodes length
		# delay * sub time slot length * propagation speed = distance
		self.delay = delay
		# the memory used to track transmitting packets
		self.sink_receiving_counter = np.zeros(self.n_nodes, dtype=int)
		self.sink_throughput_trace = np.zeros(self.n_nodes, dtype=int)
		# because src->sink delay channel can exist more than one packet
		# use the delay queue to implement it
		# object dtype numpy array is less effieicnt than python list
		# used for action arrival
		self.src2sink_delay_channels = [DELAY_QUEUE(self.delay[i]) for i in range(self.n_nodes)]
		# used for observation update
		self.sink2src_delay_channels = [DELAY_QUEUE(self.delay[i]) for i in range(self.n_nodes)]
		# self.sink2src_delay_channels = [DELAY_QUEUE(1) for i in range(self.n_nodes)] # Xuan's configuration
		# because sink should 1 time slot slower than the source
		# use it to store previous time slot observation
		self.previous_obs = -1
	
	# TODO: add the translation that returns more comprehensive info
	def __str__(self):
		return str(self.sink_throughput_trace)

	def get_trace(self):
		return self.sink_throughput_trace
	
	# API for mobility update
	# It is inefficient becuase the organization of delay queues
	def update_delay(self, new_delays: np.array):
		for i in range(self.n_nodes):
			self.src2sink_delay_channels[i].update_delay(new_delays[i])
			self.sink2src_delay_channels[i].update_delay(new_delays[i])

	# Each call of this function will be a new sub time slot
	# Input: actions-the vector denotes actions of all source nodes in the network, 0/1 for trans/not
	# Return: Observation-the broadcast contents recieved for all source nodes
	#         Because different nodes has different delay, their observations are different
	# Note: The rewards is based on the observation, it is the responsibility of nodes
	def step(self, actions: np.array, packet_length: int = 1):
		uplink_channel_output = np.array([self.src2sink_delay_channels[i].step(actions[i] * packet_length)[0] for i in range(self.n_nodes)], dtype=int)
		# in a time slot, the observation to send is the same
		# but delay are different, so different nodes can see different obs
		Observations = np.array([i.step(1, self.previous_obs)[1] for i in self.sink2src_delay_channels], dtype=int)
		uplink_channel_throughput = np.abs(uplink_channel_output)
		self.sink_receiving_counter += uplink_channel_throughput
		obs = -1
		# from the sink's perspective, success includes nodes ID info
		success_trace = np.zeros(self.n_nodes, dtype=int)
		# for uplink, check sink state
		# Observations format: [collided, idle, successful]
		if uplink_channel_throughput.sum() == 1:
			# success
			if -1 in uplink_channel_output:
				# when a whole packet finished, the packet size is the effective payload
				# currently, the trace only the cumulative result, it can be extended to step trace
				self.sink_throughput_trace[uplink_channel_output == -1] += self.sink_receiving_counter[uplink_channel_output == -1]
				self.sink_receiving_counter[uplink_channel_output == -1] = 0
				success_trace[uplink_channel_output == -1] = 1
				# only a success whole transmission lead to success observation
				obs = 2
			else:
				obs = 1
		elif uplink_channel_throughput.sum() == 0:
			# idle
			obs = 1
		else:
			# collision
			self.sink_receiving_counter[uplink_channel_throughput == 1] = 0
			obs = 0
		self.previous_obs = obs
		return Observations, success_trace

	# intelligent sink API 
	# Each call of this function will be a new sub time slot
	# Input: actions is all source nodes action
	#        broadcast is the source node ID used to broadcast
	# Return: Observation-output of uplinks (target, always only one +-1)
	#         src_instruction-source nodes listened contents
	# Note: The rewards is based on the observation, agents calculate by themselves
	def cycle(self, actions: np.array, broadcast: np.array, packet_length: int = 1):
		Observations = np.array([self.src2sink_delay_channels[i].step(actions[i] * packet_length)[0] for i in range(self.n_nodes)], dtype=int)
		abs_obs = np.abs(Observations)
		src_instruction = np.array([i.step(1, broadcast)[1] for i in self.sink2src_delay_channels], dtype=int)
		success_trace = np.zeros(self.n_nodes, dtype=int)
		self.sink_receiving_counter += abs_obs
		if -1 in Observations and abs_obs.sum() == 1:
			self.sink_throughput_trace[Observations == -1] += self.sink_receiving_counter[Observations == -1]
			self.sink_receiving_counter[Observations == -1] = 0
			success_trace[Observations == -1] = 1
		elif abs_obs.sum() > 1:
			self.sink_receiving_counter[abs_obs == 1] = 0
		return Observations, src_instruction, success_trace

# Normal nodes decision making simulation
class ENVIRONMENT(object):
	def __init__(self,
				 n_agents: int,
				 n_others: int,
				 nodes_mask: np.array,
				 packet_length: int = 1,
				 guard_length: int = 1,
				 sub_slot_length: int = 1e7,
				 frame_length: int = 10,
				 tdma_occupancy: int = 2,
				 aloha_prob: float = 0.2,
				 window_size: int = 2,
				 max_backoff: int = 2,
				 env_mode: int = 0,
				 mac_mode: int = 0,
				 sink_mode: int = 0,
				 nodes_delay: np.array = None,
				 num_sub_slot: int = 1,
				 movable: bool = False,
				 move_freq: float = 0,
				 save_trace: bool = False,
				 n_iter: int = 1000,
				 log_name: str = '',
				 config_name: str = '',
				 ):
		super(ENVIRONMENT, self).__init__()
		# Most important parameters
		# Ruoyu: agents are user-defined MAC nodes
		# if n_agents == 0, it is the pure conventional network simulation
		self.n_agents = n_agents
		self.n_others = n_others
		self.n_nodes = self.n_agents + self.n_others
		assert self.n_nodes < 1024
		# Ruoyu: Support for hybrid network
		# 0-agent, 1-TDMA, 2-qALOHA, 3-FW_ALOHA, 4-EB_ALOHA
		self.nodes_mask = nodes_mask
		assert self.nodes_mask.size == self.n_nodes
		
		# Ruoyu: simulation modes selection
		# env_mode 0: RF, env_mode others: UAN
		self.env_mode = env_mode
		# mac_mode 0: sync, mac_mode others: async
		self.mac_mode = mac_mode
		# sink_mode 0: src-agent, sink_mode other: sink-agent
		self.sink_mode = sink_mode

		self.packet_length = 0 if self.env_mode == 0 else packet_length
		self.guard_length = 0 if self.env_mode == 0 else guard_length
		self.sending_counter = np.zeros(self.n_nodes, dtype=int)
		# sub time slot length, unit in nano second (10^-9 s)
		# default 0.01 s
		self.sub_slot_length = sub_slot_length

		# conventional nodes decision making parameters
		self.node_actions = np.zeros(self.n_nodes, dtype=int)
		self.frame_length = frame_length
		self.tdma_occupancy = tdma_occupancy
		self.tdma = np.zeros(self.frame_length, dtype=int)
		self.tdma[np.random.choice(self.frame_length, self.tdma_occupancy)] = 1
		self.tdma_counter = 0
		self.aloha_prob = aloha_prob
		self.window_size = window_size
		self.max_backoff = max_backoff
		self.eb_collision_count = np.zeros(self.n_nodes, dtype=int) 
		self.window = np.random.randint(0, self.window_size * 2**self.eb_collision_count, dtype=int)

		# based on simulation mode, use different step function to ensure the APIs are the same
		self.step = self.source_agent if self.sink_mode == 0 else self.sink_agent
		# delay is the propagation delay measured by number of sub time slots 
		self.nodes_delay = np.ones(self.n_nodes, dtype=int) if self.env_mode == 0 else nodes_delay
		assert self.nodes_delay.size == self.n_nodes
		self.num_sub_slot = 1 if self.env_mode == 0 else num_sub_slot
		self.channel = CHANNEL(self.n_nodes, self.nodes_delay)
		self.n_padding = 2 * self.nodes_delay.max()

		self.previous_action = np.zeros(self.num_sub_slot, dtype=int)

		# mobile simulator need to be initlized outside
		self.movable = movable
		if self.movable:
			self.move_freq = move_freq
			
		# Ruoyu: Track the system throughput is the environment responsibility
		# it has large overhead of memory, but we need that data
		self.save_trace = save_trace
		if self.save_trace:
			self.trace_counter = 0
			self.n_iter = n_iter
			self.log_name = log_name
			self.config_name = config_name
			# the addtional rows are for the drain reward
			self.transmission_logs = np.zeros((self.n_iter + 1, self.n_nodes), dtype=int)

	# attach the mobile simulator
	def attach_spatial(self, spatial):
		self.spatial = spatial
		self.move_counter = 0
		new_distribution = self.spatial.get_distance()
		new_delay = self.distance2delay(new_distribution)
		self.update_delay(new_delay)

	def reset(self):
		self.node_actions = np.zeros(self.n_nodes, dtype=int)
		self.tdma = np.zeros(self.frame_length, dtype=int)
		self.tdma[np.random.choice(self.frame_length, self.tdma_occupancy)] = 1
		self.tdma_counter = 0
		self.eb_collision_count = np.zeros(self.n_nodes, dtype=int) 
		self.window = np.random.randint(0, self.window_size * 2**self.eb_collision_count, dtype=int)
		self.channel = CHANNEL(self.n_nodes, self.nodes_delay, self.num_sub_slot)
		self.previous_action = np.zeros(self.num_sub_slot, dtype=int)
		if self.movable:
			self.move_counter = 0		
		if self.save_trace:
			self.trace_counter = 0

	# conventional nodes decision making
	# granularity is time slot, call it at the beginning of a time slot
	# action representation is consistent between sync and async mode
	# 0: no transmission, 1: transmit at the first sub time slot, so on so forth
	def get_actions(self):
		# TDMA, always transmit at the beginning of the time slot
		self.node_actions[self.nodes_mask == 1] = self.tdma[self.tdma_counter]
		
		# q-ALOHA, sync mode transmits at the beginning, async will random choose transmit start time
		self.node_actions[self.nodes_mask == 2] = (np.random.uniform(0, 1, self.n_nodes) < self.aloha_prob)[self.nodes_mask == 2]
		if self.mac_mode:
			self.node_actions[self.nodes_mask == 2] *= np.random.randint(1, self.num_sub_slot + 1 - self.guard_length, self.n_nodes)[self.nodes_mask == 2]

		# FW/EB-ALOHA
		self.node_actions[self.nodes_mask == 3] = (self.window[self.nodes_mask == 3] == 0)
		self.node_actions[self.nodes_mask == 4] = (self.window[self.nodes_mask == 4] == 0)

	# call the function at the end of a time slot
	# this function will update all internal counters
	def update_counters(self):
		# update tdma period counter
		self.tdma_counter = (self.tdma_counter + 1) % self.frame_length
		self.window -= 1
		self.eb_collision_count = np.minimum(self.eb_collision_count, self.max_backoff)
		self.window[self.window < 0] = np.random.randint(0, self.window_size * 2**self.eb_collision_count)[self.window < 0]

	# post process observations
	# if the observation has more complex architecture
	# add the decoding part here
	def decode_obs(self, Observations):
		# treat no data as idle
		Observations[Observations == -1] = 1
		# shift [0, 1, 2] to [-1, 0, 1]
		Observations -= 1
		return Observations
	
	def distance2delay(self, distance):
		# RF propagation delay can be neglect
		if self.env_mode == 0:
			return np.ones(self.n_nodes, dtype=int)
		else:
			# distance should in unit of meter
			# UAN propagation speed 1500m/s
			# ceil rather than floor
			return -(((distance / 1500) * 1e9) // -self.sub_slot_length).astype(int)

	# for spatial initialization based on the current delay
	def delay2distance(self):
		# only the UAN needs to concern this concept
		return 1500 * (self.nodes_delay.astype(float) * self.sub_slot_length * 1e-9)

	def update_delay(self, delay):
		self.nodes_delay = delay
		self.channel.update_delay(self.nodes_delay)

	# src-agent ENV API
	# there is no more than 1 agents, the action is an integer
	# this function is compatible to non-agent simulation
	def source_agent(self, action=0):
		# get the action of this time slot
		self.get_actions()
		self.node_actions[self.nodes_mask == 0] = action
		Observations = np.zeros((self.num_sub_slot, self.n_nodes), dtype=int)
		Success_trace = np.zeros((self.num_sub_slot, self.n_nodes), dtype=int)
		for i in range(self.num_sub_slot):
			sub_slot_action = np.zeros(self.n_nodes, dtype=int)
			sub_slot_action[self.node_actions == 1] = 1
			# eliminate overlapping sending case
			sub_slot_action[self.sending_counter != 0] = 0
			self.sending_counter[sub_slot_action == 1] = self.packet_length
			# for async mode, when action == 1, it is time to go
			self.node_actions -= 1
			self.sending_counter -= 1
			self.sending_counter[self.sending_counter < 0] = 0
			# Note: obs may contain -1, represent noting
			Observations[i, :], Success_trace[i, :] = self.channel.step(sub_slot_action, self.packet_length)

			# mobility update
			if self.movable and self.move_freq != 0:
				self.move_counter += 1
				if (1 / self.move_counter) <= self.move_freq:
					self.spatial.step()
					new_distribution = self.spatial.get_distance()
					new_delay = self.distance2delay(new_distribution)
					self.update_delay(new_delay)
					self.move_counter = 0
		
		# how to use depends on the agent
		Observations = self.decode_obs(Observations)

		# update internal counters
		self.eb_collision_count[(Observations < 0).sum(0) > 0] += 1
		self.update_counters()

		if self.save_trace:
			self.transmission_logs[self.trace_counter, :] = Success_trace.sum(0)
			self.trace_counter += 1
		
		return Observations[:, self.nodes_mask == 0]

	# sink-agent ENV API
	# main part in the CHANNEL (sink)
	# ENVIRONMENT just provides APIs for agent
	# dimensions: action: # sub * 1
	#             obs: # sub * n_nodes
	#             rewards: 1 * n_nodes
	def sink_agent(self, broadcast):
		Observations = np.zeros((self.num_sub_slot, self.n_nodes), dtype=int)
		Success_trace = np.zeros((self.num_sub_slot, self.n_nodes), dtype=int)
		nodes_idx = np.array([i for i in range(self.n_nodes)], dtype=int)
		for i in range(self.num_sub_slot):
			sub_slot_action = np.zeros(self.n_nodes, dtype=int)
			sub_slot_action[nodes_idx == self.previous_action] = 1
			# eliminate overlapping sending case
			sub_slot_action[self.sending_counter != 0] = 0
			self.sending_counter[sub_slot_action == 1] = self.packet_length
			self.sending_counter -= 1
			self.sending_counter[self.sending_counter < 0] = 0
			Observations[i, :], self.previous_action, Success_trace[i, :] = self.channel.cycle(sub_slot_action, broadcast[i], self.packet_length)

		if self.save_trace:
			self.transmission_logs[self.trace_counter, :] = Success_trace.sum(0)
			self.trace_counter += 1
		
		return Observations

	# drain the delay queue
	# store logs and configs
	# return statistic throughput
	def finalize(self):
		zero_padding_actions = np.zeros(self.n_nodes, dtype=int)
		Success_trace = np.zeros((self.n_padding, self.n_nodes), dtype=int)
		nodes_idx = np.array([i for i in range(self.n_nodes)], dtype=int)
		for i in range(self.n_padding):
			if self.sink_mode == 0:
				_, Success_trace[i, :] = self.channel.step(zero_padding_actions, self.packet_length)
			else:
				zero_padding_actions[nodes_idx == self.previous_action] = 1
				zero_padding_actions[nodes_idx != self.previous_action] = 0
				_, self.previous_action, Success_trace[i, :] = self.channel.cycle(zero_padding_actions, 1023, self.packet_length)
			# mobility update
			if self.movable and self.move_freq != 0:
				self.move_counter += 1
				if (1 / self.move_counter) <= self.move_freq:
					self.spatial.step()
					new_distribution = self.spatial.get_distance()
					new_delay = self.distance2delay(new_distribution)
					self.update_delay(new_delay)
					self.move_counter = 0

		if self.movable and self.move_freq != 0:
			self.spatial.finalize()

		if self.save_trace:
			self.transmission_logs[self.trace_counter, :] = Success_trace.sum(0)
			self.trace_counter += 1
			np.savetxt(self.log_name, self.transmission_logs[:self.trace_counter, :], fmt='%d')
			with open(self.config_name, 'a') as f:
				config_list = ['n_agents', 'n_others', 'nodes_mask', 'packet_length', 'guard_length', 'frame_length', 'tdma_occupancy', 
							   'tdma', 'aloha_prob', 'window_size', 'max_backoff', 'env_mode', 'mac_mode', 'sink_mode',
							   'nodes_delay', 'num_sub_slot', 'movable', 'move_freq']
				f.write('\n======= ENVIRONMENT =======\n')
				f.write('\n'.join([f'{config_key}: {self.__dict__[config_key]}' for config_key in config_list if config_key in self.__dict__]))
				f.write('\n')
				f.close()
		return self.channel.get_trace()
