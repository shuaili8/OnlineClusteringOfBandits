import networkx as nx
import numpy as np
from utlis import fracT, edge_probability
from Base import IND

class Cluster:
	def __init__(self, users, S, b):
		self.users = users # a list/array of users
		self.S = S
		self.b = b

		# self.Sinv = np.linalg.pinv(S) # add later and theta


class CLUB(IND):
	# random_init: use random initialization or not
	# neighbor: use only information from neighbors or the connected component
	def __init__(self, nu, d, T, random_init = True, neighbor = False):
		super(CLUB, self).__init__(nu, d, T)
		self.nu = nu
		self.neighbor = neighbor

		self.N = np.zeros(self.nu) # users' appearance times
		self.alpha = 4 * np.sqrt(d) # parameter for cut edge
		self.G = nx.gnp_random_graph(self.T, edge_probability(self.T)) if random_init else nx.complete_graph(self.T)
		
		self.clusters = {0:Cluster(range(nu), np.zeros((d, d)), np.zeros(d))}
		self.cluster_inds = np.zeros(nu)

		self.num_clusters = np.ones(T)

	def recommend(self, i, items):
		cluster = self.clusters[self.cluster_inds[i]]
		kk = self._select_item_ucb(cluster.S, cluster.b, items, self.beta)

		return kk

	def store_info(self, i, x, y, t, r):
		super(CLUB, self).update(i, x, y, t, r)
		self.N[i] += 1

		c = self.cluster_inds[i]
		self.clusters[c].S += np.outer(x, x)
		self.clusters[c].b += y * x
 
	def update(self, t):
		if t % 1000 == 0:
			update_clusters = False
			for i in range(self.nu):
				if self.N[i] > 0:
					Sinv1 = np.linalg.pinv(self.S[i]) # next step is to store such Sinv's
					theta1 = np.matmul(Sinv1, self.b[i]) # next step is store theta

				A = [a for a in self.G.neighbors(i)]
				for j in A:
					if self.N[j] > 0:
						Sinv2 = np.linalg.pinv(self.S[j])
						theta2 = np.matmul(Sinv2, self.b[j])

						if np.linalg.norm(theta1 - theta2) > self.alpha * (fracT(self.N[i]) + fracT(self.N[j])):
							self.G.remove_edge(i, j)
							update_clusters = True

			if update_clusters:
				c = 0
				remain_users = set(range(self.nu))
				while len(remain_users) > 0:
					i = np.random.choice(list(remain_users))
					C = nx.node_connected_component(self.G, i)
					S = np.sum([self.S[i] for i in C])
					b = np.sum([self.b[i] for i in C])

					self.clusters[c] = Cluster(list(C), S, b)
					self.cluster_inds[np.asarray(C)] = c
					c += 1

					remain_users = remain_users - set(C)

		self.num_clusters[t] = len(self.clusters)
