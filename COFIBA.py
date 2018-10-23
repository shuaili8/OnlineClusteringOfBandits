import networkx as nx
import numpy as np
from utlis import fracT, invertible, edge_probability, select_item
from IND import IND

class COFIBA(IND):
	"""docstring for COFIBA"""
	def __init__(self, num_users, d, num_rounds, L, random_init = True):
		super(COFIBA, self).__init__(num_users, d, num_rounds)

		self.L = L

		if random_init:
			self.G = {0:nx.gnp_random_graph(num_users, edge_probability(num_users))}

			self.GI = nx.gnp_random_graph(L, edge_probability(num_users))
			c = 0
			self.i_ind = np.zeros(L)
			self.i_clusters = {}
			C = set(range(L))
			while len(C) != 0:
				l = next(iter(C))
				C0 = set(nx.node_connected_component(self.GI, l))
				self.i_clusters[c] = list(C0)
				for l1 in C0:
					self.i_ind[l1] = c
				c += 1
				C = C - C0
		else:
			self.G = {0:nx.complete_graph(num_users)}
			
			self.i_ind = np.zeros(L)
			self.i_clusters = {0:[i for i in range(L)]}
			self.GI = nx.complete_graph(L)

		self.alpha = 4 * np.sqrt(d)
		self.num_clusters = np.ones(num_rounds)

	def recommend(self, i, items):
		kk = np.zeros(len(items))
		for c in self.G:
			H = nx.node_connected_component(self.G[c], i)
			d = np.shape(self.S[0])[0]
			Sc = np.zeros((d, d))
			bc = np.zeros(d)
			Tc = 0
			for j in H:
				Sc += self.S[j]
				bc += self.b[j]
				Tc += self.T[j]

			if invertible(Sc):
				Sinv = np.linalg.inv(Sc)
				theta_est = np.dot(Sinv, bc)
				for l in self.i_clusters[c]:
					kk[l] = np.dot(items[l, :], theta_est) + self.beta * np.dot(items[l,:], np.dot(Sinv, items[l,:]))
			else:
				for l in self.i_clusters[c]:
					kk[l] = np.random.uniform(0, 1, 1)

		return np.argmax(kk)

	def find_available_index(self):
		if len(self.i_clusters) == 0:
			return 0

		cmax = max(self.i_clusters)

		for c in range(cmax + 1):
			if c not in self.i_clusters:
				return c

		return cmax + 1

	def update_item_graph(self, i, kk):
		Sinv1 = np.linalg.inv(self.S[i])
		theta1 = np.dot(Sinv1, self.b[i])

		C0 = set(nx.node_connected_component(self.GI, kk))
		# H = GI.subgraph(C0)
		A = [a for a in self.GI.neighbors(kk)]

		N0 = []
		N = [[] for l in range(len(A))]
		
		num_users = len(self.S)
		for j in range(num_users):
			if j == i:
				continue
			if invertible(self.S[j]):
				Sinv = np.linalg.inv(self.S[j])
				theta = np.dot(Sinv, self.b[j])
				if np.abs(np.dot(theta - theta1, self.items[kk, :])) < self.alpha * (fracT(self.T[i]) + fracT(self.T[j])):
					N0.append(j)
				for a in range(len(A)):
					l = A[a]
					if np.abs(np.dot(theta - theta1, self.items[l, :])) < self.alpha * (fracT(self.T[i]) + fracT(self.T[j])):
						N[a].append(j)
			else:
				N0.append(j)
				for a in range(len(A)):
					N[a].append(j)

		N0 = set(N0)
		for a in range(len(A)):
			l = A[a]
			if l == kk:
				continue
			N[a] = set(N[a])
			if N0 != N[a]:
				self.GI.remove_edge(kk, l)
				# H.remove_edge(kk, l)

		C = set(nx.node_connected_component(self.GI, kk))
		if C != C0:
			c = self.i_ind[kk]
			self.G[c] = nx.gnp_random_graph(num_users, edge_probability(num_users))
			self.i_clusters[c] = [l for l in C]
			C0 = C0 - C
			while len(C0) > 0:
				l = next(iter(C0))
				C1 = set(nx.node_connected_component(self.GI, l))
				c1 = self.find_available_index()
				self.G[c1] = nx.gnp_random_graph(num_users, edge_probability(num_users))
				self.i_clusters[c1] = [l for l in C1]
				for l1 in C1:
					self.i_ind[l1] = c1
				C0 = C0 - C1
		return

	def run(self, envir):
		self.items = envir.get_items()
		for t in range(self.num_rounds):
			if t % 5000 == 0:
				print(t // 5000, end = ' ')
			I = envir.generate_users()
			for i in I:
				kk = self.recommend(i, self.items)
				x = self.items[kk]
				y, r, br = envir.get_feedback_reward(i, kk)
				self.update(i, x, y, t, r, br)

			for i in I:
				if invertible(self.S[i]) == True:
					Sinv1 = np.linalg.inv(self.S[i])
					theta1 = np.dot(Sinv1, self.b[i])

					c = self.i_ind[kk]
					A = [a for a in self.G[c].neighbors(i)]
					for j in A:
						if invertible(self.S[j]) == True:
							Sinv2 = np.linalg.inv(self.S[j])
							theta2 = np.dot(Sinv1, self.b[j])
							if np.abs(np.dot(theta1 - theta2, self.items[kk, :])) > self.alpha * (fracT(self.T[i]) + fracT(self.T[j])):
								self.G[c].remove_edge(i, j)

					self.update_item_graph(i, kk)

			self.num_clusters[t] = sum([nx.number_connected_components(self.G[c]) for c in self.G])
		print()
		return
