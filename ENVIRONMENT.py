import numpy as np
from utlis import get_best_reward

def generate_items(num_items, d):
	x = np.random.normal(0, 1, (num_items, d))
	x = np.concatenate((np.divide(x, np.outer(np.linalg.norm(x, axis = 1), np.ones(np.shape(x)[1]))), np.ones((num_items, 1))), axis = 1)
	return x

class Environment:
	# p: frequency vector of users
	# synthetic: synthetic data / real dataset
	# fixed: fixed item set / random item set
	def __init__(self, L, d, m, num_users, p, theta, synthetic = True, fixed = False):
		self.synthetic = synthetic
		self.fixed = fixed
		self.L = L
		self.d = d
		self.p = p # probability distribution over users

		# need to rewrite the environment to return no best reward but save best reward inside environments

		self.items = generate_items(num_items = L, d = d)
		if synthetic:
			self.theta = theta

	def get_items(self):
		if self.fixed == False:
			self.items = generate_items(num_items = self.L, d = self.d)
		return self.items

	def feedback(self, i, k):
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

class Synthetic(Environment):
	def __init__(self, L, d, m, num_users, p, theta, fixed = False):
		super(Synthetic, self).__init__()
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
		
class RealData(Environment):
	"""docstring for REALDATA"""
	def __init__(self, arg):
		super(REALDATA, self).__init__()
		self.arg = arg