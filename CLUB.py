import networkx as nx
import numpy as np
from utlis import edge_probability, is_power2, isInvertible
from Base import LinUCB_IND

class Cluster:
    def __init__(self, users, S, b, N):
        self.users = users # a list/array of users
        self.S = S
        self.b = b
        self.N = N
        self.Sinv = np.linalg.inv(self.S)
        self.theta = np.matmul(self.Sinv, self.b)

class CLUB(LinUCB_IND):
    # random_init: use random initialization or not
    def __init__(self, nu, d, T, edge_probability = 1):
        super(CLUB, self).__init__(nu, d, T)
        self.nu = nu
        # self.alpha = 4 * np.sqrt(d) # parameter for cut edge
        self.G = nx.gnp_random_graph(nu, edge_probability)
        self.clusters = {0:Cluster(users=range(nu), S=np.eye(d), b=np.zeros(d), N=0)}
        self.cluster_inds = np.zeros(nu)

        self.num_clusters = np.zeros(T)

    def recommend(self, i, items, t):
        cluster = self.clusters[self.cluster_inds[i]]
        return self._select_item_ucb(cluster.S, cluster.Sinv, cluster.theta, items, cluster.N, t)

    def store_info(self, i, x, y, t, r, br):
        super(CLUB, self).store_info(i, x, y, t, r, br)

        c = self.cluster_inds[i]
        self.clusters[c].S += np.outer(x, x)
        self.clusters[c].b += y * x
        self.clusters[c].N += 1

        self.clusters[c].Sinv, self.clusters[c].theta = self._update_inverse(self.clusters[c].S, self.clusters[c].b, self.clusters[c].Sinv, x, self.clusters[c].N)

    def _if_split(self, theta, N1, N2):
        # alpha = 2 * np.sqrt(2 * self.d)
        alpha = 1
        def _factT(T):
            return np.sqrt((1 + np.log(1 + T)) / (1 + T))
        return np.linalg.norm(theta) >  alpha * (_factT(N1) + _factT(N2))
 
    def update(self, t):
        update_clusters = False
        for i in self.I:
            c = self.cluster_inds[i]

            A = [a for a in self.G.neighbors(i)]
            for j in A:
                if self.N[i] and self.N[j] and self._if_split(self.theta[i] - self.theta[j], self.N[i], self.N[j]):
                    self.G.remove_edge(i, j)
                    print(i,j)
                    update_clusters = True

        if update_clusters:
            C = set()
            for i in self.I: # suppose there is only one user per round
                C = nx.node_connected_component(self.G, i)
                if len(C) < len(self.clusters[c].users):
                    remain_users = set(self.clusters[c].users)
                    self.clusters[c] = Cluster(list(C), S=sum([self.S[k]-np.eye(self.d) for k in C])+np.eye(self.d), b=sum([self.b[k] for k in C]), N=sum([self.N[k] for k in C]))

                    remain_users = remain_users - set(C)
                    c = max(self.clusters) + 1
                    while len(remain_users) > 0:
                        j = np.random.choice(list(remain_users))
                        C = nx.node_connected_component(self.G, j)

                        self.clusters[c] = Cluster(list(C), S=sum([self.S[k]-np.eye(self.d) for k in C])+np.eye(self.d), b=sum([self.b[k] for k in C]), N=sum([self.N[k] for k in C]))
                        for j in C:
                            self.cluster_inds[j] = c

                        c += 1
                        remain_users = remain_users - set(C)

            # print(len(self.clusters))
            
        self.num_clusters[t] = len(self.clusters)

        # if t % 1000 == 0:
        #     print(self.cluster_inds)
        #     print([np.linalg.norm(self.theta[0]-self.theta[i]) for i in range(1,self.nu)])
