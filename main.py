from jacobi_eigenvalue import jacobi_eigenvalue_algorithm
from power_iteration import power_iteration
from qr_algorithm import qr_algorithm
from rayleigh_quotient_iteration import rayleigh_quotient_iteration

import numpy as np
from time import time

def main():
    # Define the input matrix
    matrix = np.array([[4, 2], [1, 3]])

    # Call the power_iteration function to compute the eigenvalue and eigenvector
    eigenvalue, eigenvector = power_iteration(matrix)

    # Print the results
    print("\nPower Iteration Results")
    print("Eigenvalue:", eigenvalue)
    print("Eigenvector:", eigenvector)

    # Call the rayleigh_quotient_iteration function to compute the eigenvalue and eigenvector
    eigenvalue, eigenvector = rayleigh_quotient_iteration(matrix)

    # Print the results
    print("\nRayleigh Quotient Iteration Results")
    print("Eigenvalue:", eigenvalue)
    print("Eigenvector:", eigenvector)

    # Call the jacobi_eigenvalue_algorithm function to compute the eigenvalue and eigenvector
    eigenvalue, eigenvector = jacobi_eigenvalue_algorithm(matrix)

    # Print the results
    print("\nJacobi Eigenvalue Algorithm Results")
    print("Eigenvalue:", eigenvalue)
    print("Eigenvector:", eigenvector)

    # Call the qr_algorithm function to compute the eigenvalue and eigenvector
    eigenvalue, eigenvector = qr_algorithm(matrix)

    # Print the results
    print("\nQR Algorithm Results")
    print("Eigenvalue:", eigenvalue)
    print("Eigenvector:", eigenvector)

if __name__ == "__main__":
    main()
    