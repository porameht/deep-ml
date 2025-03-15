def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == 'row':
        means = [sum(row) / len(row) for row in matrix]
    elif mode == 'column':
        means = [sum(col) / len(col) for col in zip(*matrix)]
    else:
        raise ValueError("Invalid mode")
    return means

# Test Case
print(calculate_matrix_mean([[1, 2, 3], 
                             [4, 5, 6]], 'row'))
print(calculate_matrix_mean([[1, 2, 3], 
                             [4, 5, 6]], 'column'))

# zip(*matrix) produces an iterator that gives tuples representing each column:
# First column: (1, 4)
# Second column: (2, 5)
# Third column: (3, 6)
