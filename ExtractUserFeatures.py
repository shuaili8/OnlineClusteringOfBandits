import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

def ExtractUserFeatures(num_users, d, filename):
    X = np.load(filename)

    A1 = X[:num_users, :]
    u, s, vt = np.linalg.svd(A1)

    u = u[:, :d-1]
    u = normalize(u, axis = 1, norm = 'l2')
    # print(np.linalg.norm(u[0,:]))

    U = np.concatenate((u, np.ones((num_users, 1))), axis = 1) / np.sqrt(2)

    return U

def kmeans_thetas(num_users, d, n_clusters, filename):
    U = np.load(filename)
    kmeans = KMeans(n_clusters=n_clusters).fit(U)
    thetas = np.zeros((num_users, d))
    for i in range(num_users):
        thetas[i] = kmeans.cluster_centers_[kmeans.labels_[i]]
    # thetas = {i:kmeans.cluster_centers_[kmeans.labels_[i]] for i in range(num_users)}
    print(thetas.shape)
    return thetas

# U = ExtractUserFeatures(num_users=1000, d=20, filename='yelp_1000user_1000item.npy')
# print(U.shape)
# np.save('yelp_1000user_d20', U)
# U = np.load('yelp_1000user_d20.npy')
# print(U)

# thetas = kmeans_thetas(num_users=1000, d=20, n_clusters=10, filename='ml_1000user_d20.npy')
# np.save('ml_1000user_d20_m10', thetas)
# thetas = np.load('ml_1000user_d20_m10.npy')
# print(thetas)

thetas = kmeans_thetas(num_users=1000, d=20, n_clusters=10, filename='yelp_1000user_d20.npy')
np.save('yelp_1000user_d20_m10', thetas)
thetas = np.load('yelp_1000user_d20_m10.npy')
print(thetas)

