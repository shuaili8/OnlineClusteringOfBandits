import numpy as np
from utlis import isInvertible
import random

class Base:
	# Base agent for online clustering of bandits
	def __init__(self, d, T):
		self.d = d
		self.T = T
		# self.beta = np.sqrt(self.d * np.log(self.T / self.d)) # parameter for select item
		self.rewards = np.zeros(self.T)
		self.best_rewards = np.zeros(self.T)

	def _beta(self, V, delta=0.001):
		return np.sqrt(np.log(np.linalg.det(V)) - 2 * np.log(delta))

	def _select_item_ucb(self, invertible, S, Sinv, theta, items):
		# items is of type np.array (L, d)
		if invertible:
			kk = np.argmax(np.dot(items, theta) + self._beta(S) * (np.matmul(items, Sinv) * items).sum(axis = 1))
		else:
			num_items = np.shape(items)[0]
			kk = random.choice(range(num_items))
		return kk

	def recommend(self, i, items):
		# items is of type np.array (L, d)
		# select one index from items to user i
		return

	def store_info(self, i, x, y, t, r, br):
		return

	def _update_inverse(self, invertible, S, b, Sinv, theta, x, t):
		if invertible:
			if t % 2000 == 0:
				Sinv = np.linalg.inv(S)
				theta = np.matmul(Sinv, b)
			else:
				temp = np.dot(Sinv, x)
				Sinv = Sinv - np.outer(temp, temp) / (1 + np.dot(x, np.dot(Sinv, x)))
				theta = np.matmul(Sinv, b)
		elif isInvertible(S):
			invertible = True
			Sinv = np.linalg.inv(S)
			theta = np.matmul(Sinv, b)
			
		return invertible, Sinv, theta

	def update(self, t):
		return

	def run(self, envir):
		for t in range(self.T):
			if t % 5000 == 0:
				print(t // 5000, end = ' ')
			self.I = envir.generate_users()
			for i in self.I:
				items = envir.get_items()
				kk = self.recommend(i, items)
				x = items[kk]
				y, r, br = envir.feedback(i, kk)
				self.store_info(i, x, y, t, r, br)

			self.update(t)

		print()

class LinUCB(Base):
	def __init__(self, d, T):
		super(LinUCB, self).__init__(d, T)
		self.S = np.zeros((d, d))
		self.b = np.zeros(d)
		self.Sinvertible = False
		self.Sinv = np.zeros((d, d))
		self.theta = np.zeros(d)

	def recommend(self, i, items):
		return self._select_item_ucb(self.Sinvertible, self.S, self.Sinv, self.theta, items)

	def store_info(self, i, x, y, t, r, br):
		self.rewards[t] += r
		self.best_rewards[t] += br

		self.S += np.outer(x, x)
		self.b += y * x

		self.Sinvertible, self.Sinv, self.theta = self._update_inverse(self.Sinvertible, self.S, self.b, self.Sinv, self.theta, x, t)

class IND(Base):
	# each user is an independent LinUCB
	def __init__(self, nu, d, T):
		super(IND, self).__init__(d, T)
		self.S = {i:np.zeros((d, d)) for i in range(nu)}
		self.b = {i:np.zeros(d) for i in range(nu)}
		self.Sinvertible = {i:False for i in range(nu)}
		self.Sinv = {i:np.zeros((d, d)) for i in range(nu)}
		self.theta = {i:np.zeros(d) for i in range(nu)}

		self.N = np.zeros(nu)

	def recommend(self, i, items):
		return self._select_item_ucb(self.Sinvertible[i], self.S[i], self.Sinv[i], self.theta[i], items)

	def store_info(self, i, x, y, t, r, br):
		self.rewards[t] += r
		self.best_rewards += br

		self.S[i] += np.outer(x, x)
		self.b[i] += y * x
		self.N[i] += 1

		self.Sinvertible[i], self.Sinv[i], self.theta[i] = self._update_inverse(self.Sinvertible[i], self.S[i], self.b[i], self.Sinv[i], self.theta[i], x, self.N[i])
