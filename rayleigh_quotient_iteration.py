import numpy as np

def rayleigh_quotient_iteration(A, max_iterations=100, tol=1e-6):
    """
    Rayleigh quotient iteration to compute the eigenvalue and eigenvector of a matrix A.

    Args:
        A (np.ndarray): The input matrix.
        max_iterations (int): The maximum number of iterations (default is 100).
        tol (float): The tolerance for convergence (default is 1e-6).

    Returns:
        float: The computed eigenvalue.
        np.ndarray: The computed eigenvector.
    """
    # Get the size of the matrix
    n = A.shape[0]

    # Initialize the eigenvector
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)

    # Iterate until convergence
    for i in range(max_iterations):
        # Compute the Rayleigh quotient
        lam = x.T @ A @ x / (x.T @ x)

        # Update the eigenvector
        x_new = np.linalg.solve(A - lam * np.eye(n), x)
        x_new = x_new / np.linalg.norm(x_new)

        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            break

        # Update the eigenvector for the next iteration
        x = x_new

    # Compute the final eigenvalue
    lam = x.T @ A @ x / (x.T @ x)

    return lam, x

# Example usage:

# Define the input matrix
A = np.array([[4, 2], [1, 3]])

# Call the rayleigh_quotient_iteration function to compute the eigenvalue and eigenvector
eigenvalue, eigenvector = rayleigh_quotient_iteration(A)

# Print the results
print("Eigenvalue:", eigenvalue)
print("Eigenvector:", eigenvector)
