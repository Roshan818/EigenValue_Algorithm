import numpy as np

def power_iteration(matrix, num_iterations=1000, epsilon=1e-8):
    """
    Power Iteration Algorithm to compute the dominant eigenvalue and eigenvector of a matrix.
    
    Args:
        matrix (np.ndarray): The input matrix.
        num_iterations (int): The number of iterations for the algorithm (default is 1000).
        epsilon (float): The convergence threshold for the eigenvalue (default is 1e-8).
        
    Returns:
        float: The dominant eigenvalue.
        np.ndarray: The eigenvector corresponding to the dominant eigenvalue.
    """
    # Get the dimensions of the input matrix
    n, m = matrix.shape
    
    # Initialize a random vector as the starting vector for power iteration
    initial_vector = np.random.rand(m)
    
    # Normalize the initial vector
    initial_vector = initial_vector / np.linalg.norm(initial_vector)
    
    # Initialize the previous eigenvalue
    prev_eigenvalue = 0
    
    # Perform power iteration
    for i in range(num_iterations):
        # Update the vector by multiplying with the matrix
        updated_vector = np.dot(matrix, initial_vector)
        
        # Compute the eigenvalue by taking the dot product of updated vector and initial vector
        eigenvalue = np.dot(updated_vector, initial_vector)
        
        # Normalize the updated vector
        updated_vector = updated_vector / np.linalg.norm(updated_vector)
        
        # Check for convergence
        if np.abs(eigenvalue - prev_eigenvalue) < epsilon:
            break
        
        # Update the initial vector and eigenvalue for the next iteration
        initial_vector = updated_vector
        prev_eigenvalue = eigenvalue
    
    # Compute the eigenvector corresponding to the dominant eigenvalue
    eigenvector = initial_vector
    
    return eigenvalue, eigenvector

# Example usage:

# Define the input matrix
matrix = np.array([[4, 2], [1, 3]])

# Call the power_iteration function to compute the eigenvalue and eigenvector
eigenvalue, eigenvector = power_iteration(matrix)

# Print the results
print("Dominant eigenvalue:", eigenvalue)
print("Eigenvector corresponding to the dominant eigenvalue:", eigenvector)
