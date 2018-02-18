from numpy import array, dot, append, matmul, flip, arange, vectorize
from numpy.linalg import inv
import matplotlib.pyplot as plt
from math import pow

"""

Degree two is a linear regression because it think it makes sense
that degree 1 is the most approximative constant which will become the
mean value and degree 2 becomes a line.

"""


def regression_matrix(points, degree=2):
    """Generate the matrix for the mean formula, supports only 2D."""
    trans_matrix = []
    solution_space = []
    for x, y in points:
        equation = []

        for deg in range(degree):
            equation.append(pow(x, deg))

        equation.reverse()
        trans_matrix.append(equation)
        solution_space.append(y)

    return array(trans_matrix), array(solution_space)


def equation_coefficients(trans_m, solution_space):
    """
    Get polynomial coefficients.

    We're looking to solve

        A*x = b
    <=> A.T * A * x = A.T * b
    <=> x = (A.T * A)^-1 * A.T * b
    """
    return inv(matmul(trans_m.T, trans_m)) @ (trans_m.T @ solution_space)

    """

    @ Operator does matrix vector multiplication, its rowwise dot-product.

    """


def equation(sol_coeffs):
    """Return solution function."""
    sp = flip(sol_coeffs, axis=0)

    constant = sp[0]
    poly_coeff = sp[1:]

    def x(x):
        """Return value."""
        s = 0
        for idx, coeff in enumerate(poly_coeff):
            s += coeff * pow(x, idx + 1)

        return s + constant

    return x


def plot(points, solution):
    """Plot solution."""
    xaxis = []
    yaxis = []

    mx = -1000000 
    mn =  1000000
    for x, y in points:
        mx = max(mx, x)
        mn = min(mn, x)
        xaxis.append(x)
        yaxis.append(y)

    plt.scatter(xaxis, yaxis)

    linear_space = arange(mn, mx, (mx - mn) / 10000)
    solution_space = vectorize(solution)

    plt.plot(linear_space, solution_space(linear_space))


def lazy_function(points, degree):
    """Lazy func."""
    (trans_m, sp) = regression_matrix(points, degree=degree)
    sol_coeffs = equation_coefficients(trans_m, sp)

    plot(points, equation(sol_coeffs))
    plt.show()


points = [(1, 2), (2, 3), (4, 3), (5, 6), (9, 0)]
lazy_function(points, 1)

