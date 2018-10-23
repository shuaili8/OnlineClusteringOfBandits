import numpy as np
from ENVIRONMENT import ENVIRONMENT
from utlis import get_best_reward

def generate_items(num_items, d):
	x = np.random.normal(0, 1, (num_items, d))
	x = np.concatenate((np.divide(x, np.outer(np.linalg.norm(x, axis = 1), np.ones(np.shape(x)[1]))), np.ones((num_items, 1))), axis = 1)
	return x

class SYNTHETIC(ENVIRONMENT):
	"""docstring for SYNTHETIC"""
	def __init__(self, L, d, m, num_users, p, theta, fixed = False):
		# super(SYNTHETIC, self).__init__()
		self.fixed = fixed
		self.L = L
		self.d = d
		self.p = p

		self.items = generate_items(num_items = L, d = d)
		self.theta = theta

	def get_items(self):
		if self.fixed == False:
			self.items = generate_items(num_items = self.L, d = self.d)
		return self.items

	def get_feedback_reward(self, i, k):
		x = self.items[k, :]
		r = np.dot(self.theta[i], x)
		y = np.random.binomial(1, r)
		br = get_best_reward(self.items, self.theta[i])
		return y, r, br

	def generate_users(self):
		X = np.random.multinomial(1, self.p)
		I = np.nonzero(X)[0]
		return I