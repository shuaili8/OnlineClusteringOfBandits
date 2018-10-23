import numpy as np
from scipy.stats import ortho_group

def generate_theta(gamma, d, m):
	a = np.random.normal(0, 1, m)
	a = a / np.linalg.norm(a)
	x = [a]
	while len(x) < m:
		a = np.random.normal(0, 1, m)
		a = a / np.linalg.norm(a)
		is_good = True
		for b in x:
			t = np.linalg.norm(a - b)
			if t > gamma + 0.01 or t < gamma - 0.01:
				is_good = False
			break
		if is_good:
			x.append(a)
			
	# # generate d = 5, gamma = 0.2, m = 5, seed = 122
	# A = np.array([np.array([ 0.3485921 , -0.35577883,  0.15251138,  0.85270425, -0.03925187]), np.array([ 0.22748511, -0.22777215,  0.22791425,  0.91356249, -0.09914152]), np.array([ 0.17991303, -0.33954924,  0.16795421,  0.90633221,  0.05187406]), np.array([ 0.29855927, -0.25398516,  0.29032017,  0.87031282,  0.06799774]), np.array([ 0.22330352, -0.39504906,  0.30325452,  0.8360592 , -0.05579867])])
	A1 = [np.concatenate((a, np.zeros(d - m))) for a in x]
	u = ortho_group.rvs(dim = d)
	thetam = np.dot(A1, u)

	return thetam
	

def get_theta_list(d, m, num_users):
	c = num_users // m
	thetam = generate_theta(gamma = 1.414, d = d, m = m)
	theta = {i:thetam[0] for i in range(c)}
	for j in range(1, m):
		theta.update({i:thetam[j] for i in range(c * j, c * (j + 1))})
	return theta