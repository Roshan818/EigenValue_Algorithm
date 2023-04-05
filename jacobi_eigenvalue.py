import numpy as np

def jacobi_eigenvalue_algorithm(A, max_iterations=100, tol=1e-6):
    """
    Jacobi eigenvalue algorithm to compute the eigenvalues and eigenvectors of a symmetric matrix A.

    Args:
        A (np.ndarray): The input symmetric matrix.
        max_iterations (int): The maximum number of iterations (default is 100).
        tol (float): The tolerance for convergence (default is 1e-6).

    Returns:
        np.ndarray: The computed eigenvalues.
        np.ndarray: The computed eigenvectors.
    """
    # Get the size of the matrix
    n = A.shape[0]

    # Initialize the eigenvalues and eigenvectors
    eigenvalues = np.diagonal(A)
    eigenvectors = np.eye(n)

    # Iterate until convergence or maximum iterations reached
    for i in range(max_iterations):
        # Find the largest off-diagonal element in magnitude
        max_idx = np.argmax(np.abs(A - np.diag(np.diagonal(A))))
        row = max_idx // n
        col = max_idx % n

        # Compute the rotation angle
        if A[row, col] != 0:
            theta = 0.5 * np.arctan(2 * A[row, col] / (A[row, row] - A[col, col]))
        else:
            theta = np.pi / 4  # If A[row, col] is zero, set theta to pi/4

        # Construct the rotation matrix
        R = np.eye(n)
        R[row, row] = np.cos(theta)
        R[col, col] = np.cos(theta)
        R[row, col] = np.sin(theta)
        R[col, row] = -np.sin(theta)

        # Update A and eigenvectors with the rotation
        A = R.T @ A @ R
        eigenvectors = eigenvectors @ R

        # Check for convergence
        off_diag_norm = np.linalg.norm(A - np.diag(np.diagonal(A)))
        if off_diag_norm < tol:
            break

    # Extract the eigenvalues from the diagonal of A
    eigenvalues = np.diagonal(A)

    return eigenvalues, eigenvectors

# Example usage:

# # Define the input symmetric matrix
# A = np.array([[4, 2, 1],
#               [2, 5, 3],
#               [1, 3, 6]])

# # Call the jacobi_eigenvalue_algorithm function to compute the eigenvalues and eigenvectors
# eigenvalues, eigenvectors = jacobi_eigenvalue_algorithm(A)

# # Print the results
# print("Eigenvalues:", eigenvalues)
# print("Eigenvectors:")
# for i in range(len(eigenvalues)):
#     print("Eigenvalue:", eigenvalues[i])
#     print("Eigenvector:", eigenvectors[:, i])
