def transpose_matrix(a: list[list[int or float]]) -> list[list[int or float]]:
    return [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]


a = [[1,2,3],[4,5,6]]
print(transpose_matrix(a))


# Transposing a matrix involves converting its rows 
# into columns and vice versa. This operation is fundamental 
# in linear algebra for various computations and transformations.

# [1 2 3]    to    [1 4]
# [4 5 6]          [2 5]
#                  [3 6]