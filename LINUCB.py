import numpy as np
from utlis import select_item
from BASE import BASE

class LINUCB(BASE):
	"""docstring for LINUCB"""
	def __init__(self, d, num_rounds):
		super(LINUCB, self).__init__(1, d, num_rounds)
		self.S = np.zeros((d, d))
		self.b = np.zeros(d)
		self.T = 0

	def recommend(self, i, items):
		kk = select_item(self.S, self.b, items, self.beta)
		return kk

	def update(self, i, x, y, t, r, br = 0):
		super(LINUCB, self).update(i, x, y, t, r, br)

		self.S += np.outer(x, x)
		self.b += y * x
		self.T += 1		