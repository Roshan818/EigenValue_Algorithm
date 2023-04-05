import numpy as np

def qr_algorithm(A, max_iterations=100, tol=1e-6):
    """
    QR algorithm to compute the eigenvalues and eigenvectors of a matrix A.

    Args:
        A (np.ndarray): The input matrix.
        max_iterations (int): The maximum number of iterations (default is 100).
        tol (float): The tolerance for convergence (default is 1e-6).

    Returns:
        np.ndarray: The computed eigenvalues.
        np.ndarray: The computed eigenvectors.
    """
    # Get the size of the matrix
    n = A.shape[0]

    # Initialize the eigenvalues and eigenvectors
    eigenvalues = np.zeros(n, dtype=complex)
    eigenvectors = np.eye(n)

    # Iterate until convergence or maximum iterations reached
    for i in range(max_iterations):
        # Compute the QR decomposition of A
        Q, R = np.linalg.qr(A)

        # Update A with RQ decomposition
        A = R @ Q

        # Update the eigenvectors
        eigenvectors = eigenvectors @ Q

        # Check for convergence
        off_diag_norm = np.linalg.norm(A - np.diag(np.diagonal(A)))
        if off_diag_norm < tol:
            break

    # Extract the eigenvalues from the diagonal of A
    eigenvalues = np.diagonal(A)

    return eigenvalues, eigenvectors

# # Example usage:

# # Define the input matrix
# A = np.array([[4, 2], [1, 3]])

# # Call the qr_algorithm function to compute the eigenvalues and eigenvectors
# eigenvalues, eigenvectors = qr_algorithm(A)

# # Print the results
# print("Eigenvalues:", eigenvalues)
# print("Eigenvectors:")
# for i in range(len(eigenvalues)):
#     print("Eigenvalue:", eigenvalues[i])
#     print("Eigenvector:", eigenvectors[:, i])
