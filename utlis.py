import numpy as np
import sys

def isInvertible(S):
	return np.linalg.cond(S) < 1 / sys.float_info.epsilon

def edge_probability(n):
	return 3 * np.log(n) / n

def is_power2(n):
	return n > 0 and ((n & (n - 1)) == 0)

def generate_items(num_items, d):
    # return a ndarray of num_items * d
    x = np.random.normal(0, 1, (num_items, d-1))
    x = np.concatenate((np.divide(x, np.outer(np.linalg.norm(x, axis = 1), np.ones(np.shape(x)[1])))/np.sqrt(2), np.ones((num_items, 1))/np.sqrt(2)), axis = 1)
    return x