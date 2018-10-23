import networkx as nx
import numpy as np
from utlis import invertible, fracT, edge_probability, select_item
from IND import IND

class CLUB(IND):
	# random_init: use random initialization or not
	# neighbor: use only information from neighbors or the connected component
	def __init__(self, num_users, d, num_rounds, random_init = True, neighbor = False):
		super(CLUB, self).__init__(num_users, d, num_rounds)

		if random_init:
			self.G = nx.gnp_random_graph(num_users, edge_probability(num_users))
		else:
			self.G = nx.complete_graph(num_users)

		self.neighbor = neighbor
		self.alpha = 4 * np.sqrt(d)
		self.num_clusters = np.ones(num_rounds)

	def recommend(self, i, items):
		if self.neighbor:
			H = self.G.neighbors(i)
		else:
			H = nx.node_connected_component(self.G, i)

		Sc = np.zeros((self.d, self.d))
		bc = np.zeros(self.d)
		for j in H:
			Sc += self.S[j]
			bc += self.b[j]

		kk = select_item(Sc, bc, items, self.beta)
		return kk

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

			for i in I:
				if invertible(self.S[i]) == True and self.T[i] > 0:
					Sinv1 = np.linalg.inv(self.S[i])
					theta1 = np.dot(Sinv1, self.b[i])

					A = [a for a in self.G.neighbors(i)]
					# A = self.G.neighbors(i)
					# the size of A will change
					for j in A:
						if invertible(self.S[j]) == True and self.T[j] > 0:
							Sinv2 = np.linalg.inv(self.S[j])
							theta2 = np.dot(Sinv1, self.b[j])
							if np.linalg.norm(theta1 - theta2) > self.alpha * (fracT(self.T[i]) + fracT(self.T[j])):
								self.G.remove_edge(i, j)

			self.num_clusters[t] = nx.number_connected_components(self.G)
		print()
		return