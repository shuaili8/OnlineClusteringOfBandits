import networkx as nx
import numpy as np
from utlis import fracT, edge_probability, is_power2, isInvertible
from Base import IND

class Cluster:
	def __init__(self, users, S, b, N):
		self.users = users # a list/array of users
		self.S = S
		self.b = b
		self.N = N
		if isInvertible(self.S):
			self.Sinvertible = True
			self.Sinv = np.linalg.inv(self.S)
			self.theta = np.matmul(self.Sinv, self.b)
		else:
			self.Sinvertible = False
			d = np.shape(self.S)[0]
			self.Sinv = np.zeros((d, d))
			self.theta = np.zeros(d)

class CLUB(IND):
	# random_init: use random initialization or not
	# neighbor: use only information from neighbors or the connected component
	def __init__(self, nu, d, T, edge_probability = 1):
		super(CLUB, self).__init__(nu, d, T)
		self.nu = nu
		# self.neighbor = neighbor
		# self.alpha = 4 * np.sqrt(d) # parameter for cut edge
		self.G = nx.gnp_random_graph(nu, edge_probability)
		self.clusters = {0:Cluster(range(nu), np.zeros((d, d)), np.zeros(d), 0)}
		self.cluster_inds = np.zeros(nu)

		self.num_clusters = np.ones(T)

	def recommend(self, i, items):
		cluster = self.clusters[self.cluster_inds[i]]
		return self._select_item_ucb(cluster.Sinvertible, cluster.S, cluster.Sinv, cluster.theta, items)

	def store_info(self, i, x, y, t, r, br):
		super(CLUB, self).store_info(i, x, y, t, r, br)

		c = self.cluster_inds[i]
		self.clusters[c].S += np.outer(x, x)
		self.clusters[c].b += y * x
		self.clusters[c].N += 1

		self.clusters[c].Sinvertible, self.clusters[c].Sinv, self.clusters[c].theta = self._update_inverse(self.clusters[c].Sinvertible, self.clusters[c].S, self.clusters[c].b, self.clusters[c].Sinv, self.clusters[c].theta, x, self.clusters[c].N)

	def _Vnorm(self, V, x):
		return np.sqrt(np.dot(x, np.dot(V, x)))
 
	def update(self, t):
		# if True: #t % 2000 == 0 or is_power2(t): # update every multiple rounds need to rewrite
		update_clusters = False
		for i in self.I: #range(self.nu):
			c = self.cluster_inds[i]
			if self.Sinvertible[i]: #isInvertible(self.S[i]): # self.N[i] > 0:
				# Sinv1 = np.linalg.pinv(self.S[i]) # next step is to store such Sinv's
				# theta1 = np.matmul(Sinv1, self.b[i]) # next step is store theta

				A = [a for a in self.G.neighbors(i)]
				for j in A:
					# print(j)
					if self.Sinvertible[j]: #isInvertible(self.S[i]): #self.N[j] > 0:
						# Sinv2 = np.linalg.pinv(self.S[j])
						# theta2 = np.matmul(Sinv2, self.b[j])

						if self._Vnorm(self.S[j], self.theta[i] - self.theta[j]) > self._beta(self.S[j]) or self._Vnorm(self.S[i], self.theta[i] - self.theta[j]) > self._beta(self.S[i]): #np.linalg.norm(theta1 - theta2) > self.alpha * (fracT(self.N[i]) + fracT(self.N[j])):
							self.G.remove_edge(i, j)
							update_clusters = True

		if update_clusters:
			C = set()
			for i in self.I: # suppose there is only one user per round
				C = nx.node_connected_component(self.G, i)
			if len(C) < len(self.clusters[c].users):
				remain_users = set(self.clusters[c].users)
				self.clusters[c] = Cluster(list(C), S=np.sum([self.S[i] for i in C]), b=np.sum([self.b[i] for i in C]), N=sum([self.N[i] for i in C]))

				remain_users = remain_users - set(C)
				c = max(self.clusters) + 1
				while len(remain_users) > 0:
					i = np.random.choice(list(remain_users))
					C = nx.node_connected_component(self.G, i)
					self.clusters[c] = Cluster(list(C), S=np.sum([self.S[i] for i in C]), b=np.sum([self.b[i] for i in C]), N=sum([self.N[i] for i in C]))
					c += 1
					remain_users = remain_users - set(C)
			# remain_users = set(range(self.nu))
			# while len(remain_users) > 0:
			# 	i = np.random.choice(list(remain_users))
			# 	C = nx.node_connected_component(self.G, i)
			# 	S = np.sum([self.S[i] for i in C])
			# 	b = np.sum([self.b[i] for i in C])

			# 	self.clusters[c] = Cluster(list(C), S, b)
			# 	self.cluster_inds[np.asarray(C)] = c
			# 	c += 1

			# 	remain_users = remain_users - set(C)

		self.num_clusters[t] = len(self.clusters)
