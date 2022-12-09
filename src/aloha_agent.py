import numpy as np

# Used for simulate an ALOHA node
# this allows the user apply different q in a network
class aloha_agent:
	def __init__(self,
				 aloha_prob=0.5,
				 mac_mode=0,
				 num_sub_slot=20,
				 guard_length=3,
				):
		self.aloha_prob = aloha_prob
		self.mac_mode = mac_mode
		self.num_sub_slot = num_sub_slot
		self.guard_length = guard_length

	# just to fit the API
	def kickoff(self):
		return np.zeros(1, dtype=int)

	def choose_action(self, state):
		action = np.random.uniform(0, 1) < self.aloha_prob
		if self.mac_mode:
			action *= np.random.randint(1, self.num_sub_slot + 1 - self.guard_length)
		return np.array([action], dtype=int)

	# just to fit the API
	def step(self, action, obs, state):
		return state

	# just to fit the API
	def learn(self):
		pass

	def finalize(self):
		pass
