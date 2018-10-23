import numpy as np
from utlis import select_item
from BASE import BASE

class IND(BASE):
	def __init__(self, num_users, d, num_rounds):
		super(IND, self).__init__(num_users, d, num_rounds)
		self.S = {i:np.zeros((d, d)) for i in range(num_users)}
		self.b = {i:np.zeros(d) for i in range(num_users)}
		self.T = np.zeros(num_users)
	
	def recommend(self, i, items):
		kk = select_item(self.S[i], self.b[i], items, self.beta)
		return kk

	def update(self, i, x, y, t, r, br = 0):
		super(IND, self).update(i, x, y, t, r, br)

		self.S[i] += np.outer(x, x)
		self.b[i] += y * x
		self.T[i] += 1