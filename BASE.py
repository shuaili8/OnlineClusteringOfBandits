import numpy as np
from utlis import select_item

class BASE(object):
	"""docstring for BASE"""
	def __init__(self, num_users, d, num_rounds):
		super(BASE, self).__init__()
		self.d = d
		self.num_rounds = num_rounds

		self.beta = np.sqrt(d * np.log(num_rounds / d))

		self.best_rewards = np.zeros(num_rounds)
		self.rewards = np.zeros(num_rounds)

	def recommend(self, i, items):
		kk = np.random.randint(num_items)
		return kk

	# def recommend_S(self, S, items)
	# 	kk = select_item(S, items, self.beta)
	# 	return kk

	def update(self, i, x, y, t, r, br = 0):
		self.rewards[t] += r
		self.best_rewards[t] += br

	def run(self, envir):
		for t in range(self.num_rounds):
			if t % 5000 == 0:
				print(t // 5000, end = ' ')
			I = envir.generate_users()
			for i in I:
				items = envir.get_items()
				kk = self.recommend(i, items)
				x = items[kk]
				y, r, br = envir.get_feedback_reward(i, kk)
				self.update(i, x, y, t, r, br)

		print()

		