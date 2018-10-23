import numpy as np
from utlis import get_best_reward

class ENVIRONMENT(object):
	# p: frequency vector of users
	# synthetic: synthetic data / real dataset
	# fixed: fixed item set / random item set
	def __init__(self, L, d, m, num_users, p, theta, synthetic = True, fixed = False):
		super(ENVIRONMENT, self).__init__()
		self.synthetic = synthetic
		self.fixed = fixed
		self.L = L
		self.d = d
		self.p = p

		self.items = generate_items(num_items = L, d = d)
		if synthetic:
			self.theta = theta

	def get_items(self):
		if self.fixed == False:
			self.items = generate_items(num_items = self.L, d = self.d)
		return self.items

	def get_feedback_reward(self, i, k):
		if self.synthetic:
			x = self.items[k, :]
			r = np.dot(self.theta[i], x)
			y = np.random.binomial(1, r)
			br = get_best_reward(self.items, self.theta[i])
			return y, r, br

	def generate_users(self):
		X = np.random.multinomial(1, self.p)
		I = np.nonzero(X)[0]
		return I
		