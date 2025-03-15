def scalar_multiply(matrix: list[list[int or float]], scalar: int or float) -> list[list[int or float]]:
    return [[element * scalar for element in row] for row in matrix]

# Test Case
print(scalar_multiply([[1, 2, 3], 
                       [4, 5, 6]], 2))

