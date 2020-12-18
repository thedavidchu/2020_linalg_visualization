"""
# lin_reg.py

## Functionality
Find the linear (or higher order) regression of an input (x) and output (y).

- Example 1:
    > Problem 6.1 from Optimization Models by Giuseppe Calafiore and Laurent El Ghaoui.

- Example 2:
    > Numbers derived from https://keisan.casio.com/exec/system/14059932254941

## TODO
- Clean code to make variables' purpose more transparent
"""

import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt


def polynomial_regression(X, Y, N: int = 1, plot: bool = False):
    """
    Regression for polynomial system of order N, given data input X and data output Y for system:
    
    X -> [SYSTEM] -> Y

    :param X: input X, 1D np.array - data input
    :param Y: input Y, 1D np.array - data output (the output from the system)
    :param N: int - order of fit
    :param plot: bool - whether to plot results or not
    :return: Output of coefficients
        - [[a, b, c, ...]].T where Y ~= ax^N + bx^N-1 + cx^N-2 + ...
    """

    if X.size != Y.size:                                        # This should be an assertion
        print(f'Wrong Lengths. X = {X.size}, Y = {Y.size}')     # This should be a warning/ assertion/ error
        return False

    n = Y.size
    Y = Y.reshape(n, 1)
    X = X.reshape(n, 1)

    # Step 1: Find A
    A = np.block([X**(i) for i in range(N, -1, -1)])
    print(f'\n>>>Step 1: Find A.\nA =\n{A}')

    # Step 2: Find A^T @ A
    print(f'\n>>>Step 2: Find A.T @ A\nA^T @ A =\n{A.T @ A}')

    # Step 3: Find inv(A^T @ A)
    print(f'\n>>>Step 3: Find inv(A^T @ A)\ninv(A^T @ A) =\n{np.linalg.inv(A.T @ A)}')

    # Step 4: Find A^T @ y
    print(f'\n>>>Step 4: Find A^T @ Y\nA^T @ Y =\n{A.T @ Y}')

    # Step 5: Solution: x* = inv(A.T @ A) @ A.T @ Y
    soln = np.linalg.inv(A.T @ A) @ A.T @ Y
    print(f'\n>>>Step 5: Solution!\nx* = inv(A.T @ A) @ A.T @ y =\n{soln}')

    # Step 6: Plot solution
    if plot:
        print('\n>>>Step 6: Plot solution')
        
        # Compute solution
        x, y = sp.symbols('x y')                            # Calculate symbolic solution
        y = 0
        for i in range(N + 1):                              # Set y = ax^N + bx^N-1 + cx^N-2 + ...
            y += soln[i, 0] * x**(N-i)
        
        print(f'Approximation = {y}')                       # Display text solution
        
        x_soln = np.linspace(np.amin(x), np.amax(x), 100)   # Compute numeric solution
        y_soln = (lambdify(x, y))(x_soln)                   # Call f(x_soln), where f = lambdify(x, y) = ax^N + bx^N-1 + cx^N-2 + ...
        
        # Plot solution and original data
        plt.figure('Linear Regression Plot')                # Create new figure with title and labels
        plt.title('Linear Regression Plot')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(X, Y, c='blue', marker='o')             # Plot original data points
        plt.plot(x_soln, y_soln, 'r-')                      # Plot approximation with least regression
        plt.show()

    return soln
    
    
if __name__ == '__main__':
    """
    x = x-coordinates
    y = y-cooridnates
    """
    
    # This is why I wish Python had a switch/case syntax outside of a dictionary
    case = 1
    
    # Example 1:
    if case == 1:
        x = np.array([[-1, 0, 1, 2]]).T
        y = np.array([[0, 0, 1, 1]]).T
    # Example 2:
    elif case == 2:
        z = np.array([[83,183],[71,168],[64,171],[69,178],[69,176],[64,172],[68,165],[59,158],[81,183],[91,182],[57,163],[65,175],[58,164],[62,175]])
        x = z[:, 0]
        y = z[:, 1]
    
    soln = linear_regression(x, y, N = 2, plot=True)
