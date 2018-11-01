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

