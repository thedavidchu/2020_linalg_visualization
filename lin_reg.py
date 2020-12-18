import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt


def linear_regression(x, y, N: int = 1, plot: bool = False):
    """
    Linear regression for LINEAR system.

    :param x: input x, 1D np.array
    :param y: input y, 1D np.array
    :param N: order of fit
    :param plot: bool - whether to plot results or not
    :return: Output X
    """

    if x.size != y.size:
        print(f'Wrong Lengths. x = {x.size}, y = {y.size}')
        return False

    n = y.size
    y = y.reshape(n, 1)
    x = x.reshape(n, 1)

    # Step 1: Find A
    A = np.block([x**(i) for i in range(N, -1, -1)])
    print(f'\n>>>Step 1: Find A.\nA =\n{A}')

    # Step 2: Find A^T @ A
    print(f'\n>>>Step 2: Find A.T @ A\nA^T @ A =\n{A.T@A}')

    # Step 3: Find inv(A^T @ A)
    print(f'\n>>>Step 3: Find inv(A^T @ A)\ninv(A^T @ A) =\n{np.linalg.inv(A.T@A)}')

    # Step 4: Find A^T @ y
    print(f'\n>>>Step 4: Find A^T @ y\nA^T @ y =\n{A.T @ y}')

    # Step 5: Solution: x* = inv(A.T @ A) @ A.T @ y
    soln = np.linalg.inv(A.T @ A) @ A.T @ y
    print(f'\n>>>Step 5: Solution!\nx* = inv(A.T @ A) @ A.T @ y =\n{soln}')

    # Step 6: Plot
    if plot:
        # Compute solution
        X, Y = sp.symbols('x y')
        Y = 0
        for i in range(N+1):
            Y += soln[i, 0]*X**(N-i)
        print(Y)
        f = lambdify(X, Y)
        x_soln = np.linspace(np.amin(x), np.amax(x), 100)
        y_soln = f(x_soln)
        
        # Plot
        plt.figure('Linear Regression Plot')
        plt.title('Linear Regression Plot')
        plt.xlabel('x')
        plt.ylabel('y')
        # Plot points
        plt.scatter(x, y, c='blue', marker='o')
        # Plot line
        plt.plot(x_soln, y_soln, 'r-')
        plt.show()

    return soln
    
    
if __name__ == '__main__':
    """
    x = x-coordinates
    y = y-cooridnates
    """
    #x = np.array([[-1, 0, 1, 2]]).T
    #y = np.array([[0, 0, 1, 1]]).T
    z = np.array([[83,183],[71,168],[64,171],[69,178],[69,176],[64,172],[68,165],[59,158],[81,183],[91,182],[57,163],[65,175],[58,164],[62,175]])

    x = z[:, 0]
    y = z[:, 1]
    
    soln = linear_regression(x, y, N = 2, plot=True)

    # input('Press ENTER to continue')
