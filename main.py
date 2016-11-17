import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x * math.log(x + 2) ** 2


def legendre(x, n):
    """
    calculate nth legendre polynomial in x
    """
    if n == 0:
        return 1
    elif n == 1:
        return x
    elif n > 1:
        return ((2 * (n - 1) + 1) * x * legendre(x, n - 1) - (n - 1) * legendre(x, n - 2)) / n
    else:
        raise ValueError


def legendre_val(ps, x):
    """
    calculate values of legendre polynomials in x with coefficients ps
    """
    power, = ps.shape
    return sum(legendre(x, i)*p for i, p in enumerate(ps))


# starting values
start, stop = -1, 1
POLYPOWER = 3  # power of polynomial
xs = np.linspace(start, stop, 5)
ys = np.array([f(x) for x in xs])


# for f(xs) = ys
# a - coefficients of aproximated polynomial
# they are solution of
# Ha = b
# where
# Q = matrix of polynomial elements phi_i (i=1,m) in points x_j (j=1,n)
# H = Q.T @ Q, b = Q.T @ ys


# task 1
Q = np.array([[x**i for i in range(POLYPOWER+1)] for x in xs])
H = Q.T @ Q
b = Q.T @ ys
a = np.linalg.solve(H, b)

# task 2
lQ = np.array([[legendre(x, i) for i in range(POLYPOWER+1)] for x in xs])
lH = lQ.T @ lQ
lb = lQ.T @ ys
la = np.linalg.solve(lH, lb)

print(Q)
print(Q @ a)
print(lQ)
print(lQ @ la)

# plotting
ax = plt.figure().add_subplot(1, 1, 1)
ax.set_title('Aproximation')
plot_xs = np.linspace(start, stop)
ax.plot(plot_xs, [f(x) for x in plot_xs], label='f(x)')
ax.plot(plot_xs, [np.polyval(a[::-1], x) for x in plot_xs], label='polynomial')
ax.plot(plot_xs, [legendre_val(la, x) for x in plot_xs], label='legendre poly')
ax.legend()


plt.show()
