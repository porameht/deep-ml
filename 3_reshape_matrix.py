import numpy as np

def reshape_matrix(a: list[list[int or float]], new_shape: tuple[int, int]) -> list[list[int or float]]:
    #Write your code here and return a python list after reshaping by using numpy's tolist() method
    if len(a) == 0 or len(a[0]) == 0:
        return []
    
    # Calculate the total number of elements in the input matrix
    total_elements = sum(len(row) for row in a)
    
    print(total_elements,'total_elements')
    
    # Calculate the total number of elements required by the new shape
    required_elements = new_shape[0] * new_shape[1]
    
    print(required_elements,'required_elements')
    
    # If the shapes don't match, return an empty list
    if total_elements != required_elements:
        return []
    
    # Reshape the matrix if dimensions are compatible
    print(np.reshape(a, new_shape).tolist(),'reshaped_matrix')
    return np.reshape(a, new_shape).tolist()

# Test Case
print(reshape_matrix([[1,2,3,4],[5,6,7,8]], (4, 2)))