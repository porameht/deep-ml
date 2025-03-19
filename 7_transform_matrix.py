import numpy as np

def transform_matrix(A: list[list[int or float]], T: list[list[int or float]], S: list[list[int or float]]) -> list[list[int or float]]:
	# Convert input lists to numpy arrays for matrix operations
	A = np.array(A)
	T = np.array(T)
	S = np.array(S)
	
	if abs(np.linalg.det(S)) < 1e-8:
		return -1
	
	# Calculate inverse of T
	T_inv = np.linalg.inv(T)
	
	# Transform by T inverse first
	result = np.dot(A, T_inv)
	# Then transform by S
	result = np.dot(result, S)
	
	return result.tolist()

# Test with the example
A = [[1, 2], [3, 4]]
T = [[2, 0], [0, 2]]
S = [[1, 1], [0, 1]]
print(transform_matrix(A, T, S))