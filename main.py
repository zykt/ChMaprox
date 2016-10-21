import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x * math.log(x + 2) ** 2


# start values
start, stop = -1, 1
POLYPOWER = 3  # power of polynomial

xs = np.linspace(start, stop, 5)
ys = np.array([f(x) for x in xs])
Q = np.array([[x**i for i in range(POLYPOWER+1)] for x in xs])
H = Q.T @ Q
b = Q.T @ ys
a = np.linalg.solve(H, b)


# plotting
ax = plt.figure().add_subplot(1, 1, 1)
plot_xs = np.linspace(start, stop)
ax.plot(plot_xs, [f(x) for x in plot_xs], label='f(x)')
ax.plot(plot_xs, [np.polyval(a[::-1], x) for x in plot_xs], label='polynomial')
ax.legend()

plt.show()
