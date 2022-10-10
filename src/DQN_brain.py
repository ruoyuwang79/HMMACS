import os
import numpy as np

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Add, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.activations import softmax

class DQN:
	def __init__(self, 
				 state_len,
				 n_nodes,
				 num_sub_slot,
				 guard_length=1,
				 memory_size=500,
				 replace_target_iter=200,
				 batch_size=32,
				 learning_rate=0.01,
				 gamma=0.9,
				 epsilon=1,
				 epsilon_min=0.01,
				 epsilon_decay=0.995,
				 alpha=1,
				 mac_mode=0,
				 sink_mode=0,
				 reward_polarity=False,
				 penalty_factor=1,
				 save_trace=False,
				 config_name=''
				 ):
		# number of past time slot states in the state
		self.state_len = state_len
		# include self
		self.n_nodes = n_nodes
		# number of sub slots in one time slot, determines the action max
		self.num_sub_slot = num_sub_slot
		# guard time length (in # of sub time slot)
		self.guard_length = guard_length
		# controls
		self.mac_mode = mac_mode
		self.sink_mode = sink_mode
		self.reward_polarity = reward_polarity
		
		# shapes
		# number of actions in async mode is any possible sub time slot + idle
		self.n_actions = (self.num_sub_slot - self.guard_length if self.mac_mode == 0 else 1) + 1
		self.n_actions = self.n_actions if self.sink_mode == 0 else self.n_nodes + 1
		# action shape: action_dim1 (batch size) * action_dim2
		self.action_dim2 = 1 if self.sink_mode == 0 else self.num_sub_slot
		# observation shape: observation cannot reduce, therefore, it contains sub slot factor
		self.obs_dim2 = self.num_sub_slot if self.sink_mode == 0 else self.n_nodes * self.num_sub_slot
		# reward shape: reward_dim1 (batch size) * reward_dim2
		self.reward_dim2 = 1 if self.sink_mode == 0 else self.n_nodes
		# time slot state: s = (a, obs, R), obs in sub time slot
		self.time_slot_state_size = self.action_dim2 + self.obs_dim2 + self.reward_dim2
		# state: S = state_len * s
		self.state_size = self.time_slot_state_size * self.state_len
		# number of stored past stats
		self.memory_size = memory_size
		# memory state: (S, a, R, S_)
		self.memory = np.zeros((self.memory_size, self.state_size * 2 + (self.action_dim2 + self.reward_dim2)), dtype=int) 
		
		# frequency of target network update
		self.replace_target_iter = replace_target_iter
		# batch size of NN training input
		self.batch_size = batch_size
		# learning rate of NN training
		self.learning_rate = learning_rate
		# discount factor of Q learning
		self.gamma = gamma
		# epsilon-greedy of Q learning
		self.epsilon = epsilon
		# minimum value of epsilon
		self.epsilon_min = epsilon_min
		# epsilon discount factor
		self.epsilon_decay = epsilon_decay  
		# learning of Q learning 
		self.alpha = alpha
		# penalty factor of the collision event
		self.penalty_factor = penalty_factor
		
		# internal counters
		self.learn_step_counter = 0
		self.memory_couter = 0
				
		# build mode
		self.model        = self.build_ResNet_model() # model: evaluate Q value
		self.target_model = self.build_ResNet_model() # target_mode: target network

		# save config options
		self.save_trace = save_trace
		self.config_name = config_name

	def build_ResNet_model(self):
		# DLMA design, the flatten input
		inputs = Input(shape=(self.state_size, ))
		h1 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=247))(inputs) #h1
		h2 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=2407))(h1) #h2

		h3 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=2403))(h2) #h3
		h4 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=24457))(h3) #h4
		add1 = Add()([h4, h2])
		
		h5 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=24657))(add1) #h5
		h6 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=27567))(h5) #h6
		add2 = Add()([h6, add1])

		h7 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=24657))(add2) #h5
		h8 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=27567))(h7) #h6
		add3 = Add()([h7, add2])

		# the output is how many actions to take this time slot (sink will takes # sub slots actions) *
		# number of actions (expectations of different action) * number of observable rewards (sink see all)
		outputs =  Dense(self.action_dim2 * self.n_actions * self.reward_dim2, kernel_initializer=he_normal(seed=27))(add3)
		model = Model(inputs=inputs, outputs=outputs)
		model.compile(loss='mse', optimizer=RMSprop(learning_rate=self.learning_rate))
		return model
	
	# use Q value to determine action
	def get_action(self, action_values):
		# reshape the flatten output and get the sum up expectation of each action
		action_values_list = action_values.reshape((-1, self.action_dim2, self.n_actions, self.reward_dim2)).sum(3)
		# get the action that maximize the system throughput
		return action_values_list.argmax(2)

	# inference, batch size always 1
	def choose_action(self, state):
		self.epsilon *= self.epsilon_decay
		self.epsilon  = max(self.epsilon_min, self.epsilon)
		
		# epsilon greedy
		if np.random.uniform(0, 1) < self.epsilon:
			return np.random.randint(0, self.n_actions, self.action_dim2)

		# construct as batch 1
		state = state[np.newaxis, :]
		action_values = self.model.predict(state, verbose=0)[0]
		return self.get_action(action_values)[0]

	# startup state
	def kickoff(self):
		return np.zeros(self.state_size)

	# based on received observation, calculate rewards
	# obs: -1 collision, 0 idle, 1 success
	# the idx=0 in the first dim of the obs should be agent
	# TODO: more advanced reward function
	# sink agent case, the observation is quite different
	def get_rewards(self, obs):
		if self.sink_mode == 0:
			obs[obs == -1] = -1 * self.penalty_factor if self.reward_polarity else 0
			return np.array([obs.sum()], dtype=int)
		else:
			return (obs[np.abs(obs).sum(1) == 1] == -1).sum(0)

	# update memory
	def store_transition(self, s, a, R, s_):
		transition = np.concatenate((s, a, R, s_))
		# the rightmost may not the newest
		index = self.memory_couter % self.memory_size
		self.memory[index, :] = transition
		self.memory_couter   += 1

	# use the (obs, a) pair to construct next state, update memory
	def step(self, action, obs, state):
		Rewards = self.get_rewards(obs)
		next_state = np.concatenate((state[self.time_slot_state_size:], action, obs.flatten(), Rewards))
		self.store_transition(state, action, Rewards, next_state)
		return next_state

	# update target network
	def repalce_target_parameters(self):
		weights = self.model.get_weights()
		self.target_model.set_weights(weights)

	# DQN training
	def learn(self):
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.repalce_target_parameters()
		self.learn_step_counter += 1

		if self.memory_couter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_couter, size=self.batch_size)        
		batch_memory = self.memory[sample_index, :]

		state      = batch_memory[:, :self.state_size]
		action     = batch_memory[:, self.state_size:self.state_size + self.action_dim2]
		Rewards    = batch_memory[:, self.state_size + self.action_dim2:self.state_size + self.action_dim2 + self.reward_dim2]
		next_state = batch_memory[:, -self.state_size:]

		# q(S_t, A)
		q = self.model.predict(state, verbose=0)
		# q(S'_t, A)
		q_targ = self.target_model.predict(next_state, verbose=0)
		action_ = self.get_action(q_targ)

		# Bellman equation
		batch_idx = np.array([i for i in range(self.batch_size)], dtype=int)
		sub_slot_idx = np.array([i for i in range(self.action_dim2)], dtype=int)
		if self.sink_mode == 0:
			q.reshape((-1, self.action_dim2, self.n_actions, self.reward_dim2))[batch_idx, sub_slot_idx, action, :] = (1 - self.alpha) * q.reshape((-1, self.action_dim2, self.n_actions, self.reward_dim2))[batch_idx, sub_slot_idx, action, :] \
																				                                	+ self.alpha * (Rewards + self.gamma * q_targ.reshape((-1, self.action_dim2, self.n_actions, self.reward_dim2))[batch_idx, sub_slot_idx, action_, :])
		else:
			for i in range(self.action_dim2):
				q.reshape((-1, self.action_dim2, self.n_actions, self.reward_dim2))[batch_idx, i, action[:, i], :] = (1 - self.alpha) * q.reshape((-1, self.action_dim2, self.n_actions, self.reward_dim2))[batch_idx, i, action[:, i], :] \
																												   + self.alpha * (Rewards + self.gamma * q_targ.reshape((-1, self.action_dim2, self.n_actions, self.reward_dim2))[batch_idx, i, action_[:, i], :])

		# internal NN training
		self.model.fit(state, q, self.batch_size, epochs=1, verbose=0)

	# store configs
	def finalize(self):
		if self.save_trace:
			with open(self.config_name, 'a') as f:
				config_list = ['state_len', 'n_nodes', 'num_sub_slot', 'n_actions', 'mac_mode', 'sink_mode', 'reward_polarity',
							   'reward_dim2',  'time_slot_state_size', 'memory_size', 'replace_target_iter', 'batch_size', 'learning_rate',
							   'gamma', 'epsilon', 'epsilon_min', 'epsilon_decay', 'alpha', 'penalty_factor']
				f.write('\n======= DQN model =======\n')
				f.write('\n'.join([f'{config_key}: {self.__dict__[config_key]}' for config_key in config_list]))
				f.write('\n')
				f.close()
