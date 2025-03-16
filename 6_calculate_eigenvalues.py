def calculate_eigenvalues(matrix: list[list[float or int]]) -> list[float]:
    # Check if matrix is 2×2
    if len(matrix) != 2 or len(matrix[0]) != 2:
        raise ValueError("This implementation only supports 2×2 matrices")
    
    # Extract matrix elements
    a, b = matrix[0]
    c, d = matrix[1]
    
    # Calculate trace (sum of diagonal elements)
    trace = a + d
    
    # Calculate determinant
    determinant = a * d - b * c
    
    # Calculate discriminant
    discriminant = trace**2 - 4 * determinant
    
    # Calculate eigenvalues using the quadratic formula
    if discriminant >= 0:
        sqrt_discriminant = discriminant**0.5
        eigenvalue1 = (trace + sqrt_discriminant) / 2
        eigenvalue2 = (trace - sqrt_discriminant) / 2
        eigenvalues = [eigenvalue1, eigenvalue2]
    else:
        # Complex eigenvalues (returning real parts only)
        real_part = trace / 2
        eigenvalues = [real_part, real_part]
        # Note: For complete solution, return complex numbers
    
    return eigenvalues

# Test with the example
matrix = [[2, 1], [1, 2]]
print(calculate_eigenvalues(matrix))  # Expected output: [3.0, 1.0]