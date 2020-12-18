"""
# Plot in 3D

## Functionality

Plot 3D image of a given function on a given domain.

## TODO

Add sympy functionality for derivatives.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ============================== INPUT VARIABLES ============================== #

f = lambda x, y: x * y
domx = [-10, 10]
domy = [-10, 10]
stepx = 100
stepy = 100

# ============================== WORKING CODE ============================== #
def plot3d(f, domx, domy, stepx: int = 100, stepy: int = 100):
    """
    Plot f in 3D over domain with number of steps.

    :param f: function pointer
    :param domx: list, len(2) - [min(x), max(x)]
    :param domy: list, len(2) - [min(y), max(y)]
    :param stepx: (optional) int = 100 - number of steps on x-axis
    :param stepy: (optional) int = 100 - number of steps on y-axis
    :return: X, Y, Z
    """
    
    # Do mathematics
    X, Y = np.linspace(domx[0], domx[1], stepx), np.linspace(domy[0], domy[1], stepy)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)

    # Create 3D figure object
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('3D Plot')

    # Set X-Y-Z labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot X-Y-Z
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(np.amin(Z), np.amax(Z))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    return X, Y, Z

if __name__ == '__main__':
    X, Y, Z = plot3d(f, domx, domy, stepx, stepy)
