import numpy as np
from utlis import isInvertible
from Base import LinUCB_IND

class Cluster:
    def __init__(self, users, S, b, N, checks):
        self.users = users # a list/array of users
        self.S = S
        self.b = b
        self.N = N
        self.checks = checks
        
        self.Sinv = np.linalg.inv(self.S)
        self.theta = np.matmul(self.Sinv, self.b)
        self.checked = len(self.users) == sum(self.checks.values())

    def update_check(self, i):
        self.checks[i] = True
        self.checked = len(self.users) == sum(self.checks.values())
        
class SCLUB(LinUCB_IND):
    def __init__(self, nu, d, num_stages):
        super(SCLUB, self).__init__(nu, d, 2 ** num_stages - 1)

        self.clusters = {0:Cluster(users=[i for i in range(nu)], S=np.eye(d), b=np.zeros(d), N=0, checks={i:False for i in range(nu)})}
        self.cluster_inds = np.zeros(nu)

        self.num_stages = num_stages
        # self.alpha = 4 * np.sqrt(d)
        # self.alpha_p = np.sqrt(4) # 2
        self.num_clusters = np.ones(self.T)

    def _init_each_stage(self):
        for c in self.clusters:
            self.clusters[c].checks = {i:False for i in self.clusters[c].users}
            self.clusters[c].checked = False
            
    def recommend(self, i, items, t):
        cluster = self.clusters[self.cluster_inds[i]]
        return self._select_item_ucb(cluster.S, cluster.Sinv, cluster.theta, items, cluster.N, t)

    def store_info(self, i, x, y, t, r, br = 1):
        super(SCLUB, self).store_info(i, x, y, t, r, br)

        c = self.cluster_inds[i]
        self.clusters[c].S += np.outer(x, x)
        self.clusters[c].b += y * x
        self.clusters[c].N += 1

        self.clusters[c].Sinv, self.clusters[c].theta = self._update_inverse(self.clusters[c].S, self.clusters[c].b, self.clusters[c].Sinv, x, self.clusters[c].N)

    def _factT(self, T):
            return np.sqrt((1 + np.log(1 + T)) / (1 + T))

    def _split_or_merge(self, theta, N1, N2, split=True):
        # alpha = 2 * np.sqrt(2 * self.d)
        alpha = 1
        if split:
            return np.linalg.norm(theta) >  alpha * (self._factT(N1) + self._factT(N2))
        else:
            return np.linalg.norm(theta) <  alpha * (self._factT(N1) + self._factT(N2)) / 2

    def _cluster_avg_freq(self, c, t):
        return self.clusters[c].N / (len(self.clusters[c].users) * t)

    def _split_or_merge_p(self, p1, p2, t, split=True):
        # def _pradius(t):
        #     return np.sqrt((np.log(4) + 4 * np.log(t)) / (2*t))
        alpha_p = np.sqrt(2)
        if split:
            return np.abs(p1-p2) > alpha_p * self._factT(t)
        else:
            return np.abs(p1-p2) < alpha_p * self._factT(t) / 2

    def split(self, i, t):
        c = self.cluster_inds[i]
        cluster = self.clusters[c]

        cluster.update_check(i)

        if self._split_or_merge_p(self.N[i]/(t+1), self._cluster_avg_freq(c, t+1), t+1, split=True) or self._split_or_merge(self.theta[i] - cluster.theta, self.N[i], cluster.N, split=True):

            def _find_available_index():
                cmax = max(self.clusters)
                for c1 in range(cmax + 1):
                    if c1 not in self.clusters:
                        return c1
                return cmax + 1

            cnew = _find_available_index()
            self.clusters[cnew] = Cluster(users=[i],S=self.S[i],b=self.b[i],N=self.N[i],checks={i:True})
            self.cluster_inds[i] = cnew

            cluster.users.remove(i)
            cluster.S = cluster.S - self.S[i] + np.eye(self.d)
            cluster.b = cluster.b - self.b[i]
            cluster.N = cluster.N - self.N[i]
            del cluster.checks[i]

    def merge(self, t):
        cmax = max(self.clusters)

        for c1 in range(cmax + 1):
            if c1 not in self.clusters or self.clusters[c1].checked == False:
                continue

            for c2 in range(c1 + 1, cmax + 1):
                if c2 not in self.clusters or self.clusters[c2].checked == False:
                    continue

                if self._split_or_merge(self.clusters[c1].theta - self.clusters[c2].theta, self.clusters[c1].N, self.clusters[c2].N, split=False) and self._split_or_merge_p(self._cluster_avg_freq(c1, t+1), self._cluster_avg_freq(c2, t+1), t+1, split=False):

                    for i in self.clusters[c2].users:
                        self.cluster_inds[i] = c1

                    self.clusters[c1].users = self.clusters[c1].users + self.clusters[c2].users
                    self.clusters[c1].S = self.clusters[c1].S + self.clusters[c2].S - np.eye(self.d)
                    self.clusters[c1].b = self.clusters[c1].b + self.clusters[c2].b
                    self.clusters[c1].N = self.clusters[c1].N + self.clusters[c2].N
                    self.clusters[c1].checks = {**self.clusters[c1].checks, **self.clusters[c2].checks}

                    del self.clusters[c2]

    def run(self, envir):
        for s in range(self.num_stages):
            print(s, end = ' ')
            for t in range(2 ** s):
                if t % 5000 == 0:
                    print(t // 5000, end = ' ')

                self._init_each_stage()
                tau = 2 ** s + t - 1

                I = envir.generate_users()
                for i in I:
                    items = envir.get_items()
                    kk = self.recommend(i, items, tau)
                    x = items[kk]
                    y, r, br = envir.feedback(i, kk)
                    self.store_info(i, x, y, tau, r, br)

                for i in I:
                    # c = self.cluster_inds[i]
                    self.split(i, tau)

                self.merge(tau)
                self.num_clusters[tau] = len(self.clusters)

        print()

        