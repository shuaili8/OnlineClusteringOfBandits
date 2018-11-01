import numpy as np

class Base:
	# Base agent for online clustering of bandits
	def __init__(self, d, T):
		# self.d = d # need it or not?
		self.T = T

		self.beta = np.sqrt(self.d * np.log(self.T / self.d)) # parameter for select item
		self.rewards = np.zeros(self.T)

	def _select_item_ucb(self, S, b, items, beta):
		# items is of type np.array (L, d)
		num_items = np.shape(items)[0]
		Sinv = np.linalg.pinv(S) # change to update Sinv every ... rounds
		theta = np.matmul(Sinv, b)
		kk = np.argmax(np.dot(self.items, theta) + beta * (np.matmul(self.items, Sinv) * self.items).sum(axis = 1))

		return kk

	def recommend(self, i, items):
		# select one index from items to user i
		return

	def store_info(self, i, x, y, t, r):
		return

	def update(self, t):
		return

	def run(self, envir):
		for t in range(self.T):
			if t % 5000 == 0:
				print(t // 5000, end = ' ')
			I = envir.generate_users()
			for i in I:
				items = envir.get_items()
				kk = self.recommend(i, items)
				x = items[kk]
				y, r = envir.feedback(i, kk)
				self.store_info(i, x, y, t, r)

			self.update(t)

		print()

class LinUCB(Base):
	def __init__(self, d, T):
		super(LINUCB, self).__init__(d, T)
		self.S = np.zeros((d, d))
		self.b = np.zeros(d)

	def recommend(self, i, items):
		return self._select_item_ucb(self.S, self.b, items, self.beta)

	def store_info(self, i, x, y, t, r, br = 0):
		self.rewards[t] += r

		self.S += np.outer(x, x)
		self.b += y * x	

class IND(Base):
	# each user is an independent LinUCB
	def __init__(self, nu, d, T):
		super(IND, self).__init__(d, T)
		self.S = {i:np.zeros((d, d)) for i in range(nu)}
		self.b = {i:np.zeros(d) for i in range(nu)}
		# self.N = np.zeros(nu) # users' appearance times

	def recommend(self, i, items):
		return self._select_item_ucb(self.S[i], self.b[i], items, self.beta)

	def store_info(self, i, x, y, t, r):
		self.rewards[t] += r

		self.S[i] += np.outer(x, x)
		self.b[i] += y * x
		# self.N[i] += 1


