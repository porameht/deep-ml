def matrix_dot_vector(a: list[list[int or float]], b: list[int or float]) -> list[int or float]:
    if len(a[0]) != len(b):
        return -1
    
    result = []
    for row in a:
        total = 0
        for i in range(len(row)):
            total += row[i] * b[i]
        result.append(total)
    return result

print(matrix_dot_vector([[1, 2, 3], [2, 4, 5], [6, 8, 9]], [1, 2, 3]))

# explanation:
#  a = [[1, 2, 3],  b = [1, 2, 3]
#       [2, 4, 5], 
#       [6, 8, 9]]

# row 1: 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
# row 2: 2*1 + 4*2 + 5*3 = 2 + 8 + 15 = 25
# row 3: 6*1 + 8*2 + 9*3 = 6 + 16 + 27 = 49

#  result = [1*1 + 2*2 + 3*3, 
#            2*1 + 4*2 + 5*3, 
#            6*1 + 8*2 + 9*3]

#  result = [14, 25, 49]


