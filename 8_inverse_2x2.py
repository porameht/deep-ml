def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
	# Calculate determinant
	determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
	
	# Check if matrix is invertible (determinant != 0)
	if determinant == 0:
		return None
	
	# Calculate inverse using the formula: (1/det) * [d, -b; -c, a]
	factor = 1 / determinant
	return [
		[factor * matrix[1][1], factor * (-matrix[0][1])],
		[factor * (-matrix[1][0]), factor * matrix[0][0]]
	]

matrix = [[4, 7], [2, 6]]

print(inverse_2x2(matrix))
