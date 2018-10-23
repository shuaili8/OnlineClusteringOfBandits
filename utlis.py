import numpy as np
import sys

def get_best_reward(items, theta):
	return np.max(np.dot(items, theta))

def fracT(T):
	if T == 0:
		return 0
	return np.sqrt(np.log(T)) / T

def invertible(S):
	return np.linalg.cond(S) < 1 / sys.float_info.epsilon

def edge_probability(n):
	return 3 * np.log(n) / n

def select_item(S, b, items, beta):
	num_items = np.shape(items)[0]
	
	if invertible(S):
		Sinv = np.linalg.inv(S)
		theta_est = np.dot(Sinv, b)
		kk = np.argmax(np.dot(items, theta_est) + [beta * np.dot(items[k,:], np.dot(Sinv, items[k,:])) for k in range(num_items)])
	else:
		kk = np.random.randint(num_items)

	return kk