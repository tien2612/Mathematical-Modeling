
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv


### a = -2, b = 4, c = -1, d = -2
### R0= 2, J0= -4

def S1(y1, y2, y1_new, y2_new, h):
    return y1 + h * ((-2*y1_new) + (4*y2_new)) - y1_new


def S2(y1, y2, y1_new, y2_new, h):
    return y2 + h * ((-1*y1_new) + (-2*y2_new)) - y2_new


def jacobian(y1, y2, y1_new, y2_new, h):
    J = np.ones((2, 2))
    d = 1e-9

    J[0, 0] = (S1(y1, y2, (y1_new + d), y2_new, h) - S1(y1, y2, y1_new, y2_new, h)) / d
    J[0, 1] = (S1(y1, y2, y1_new, (y2_new + d), h) - S1(y1, y2, y1_new, y2_new, h)) / d

    J[1, 0] = (S2(y1, y2, (y1_new + d), y2_new, h) - S2(y1, y2, y1_new, y2_new, h)) / d
    J[1, 1] = (S2(y1, y2, y1_new, (y2_new + d), h) - S2(y1, y2, y1_new, y2_new, h)) / d

    return J


def NewtonRaphson(y1, y2, y1_guess, y2_guess, h):
    S_new = np.ones((2, 1))
    S_old = np.ones((2, 1))
    S_old[0] = y1_guess
    S_old[1] = y2_guess

    F = np.ones((2, 1))
    aplha = 1
    error = 9e9
    tol = 1e-9

    iter = 1
    Jold = np.ones((2, 2))

    while error > tol:

        Jnew = jacobian(y1, y2, S_old[0], S_old[1], h)
        if ((np.linalg.det(Jold) == np.linalg.det(Jnew)) or (np.linalg.det(Jnew) == 0)):
            break
        F[0] = S1(y1, y2, S_old[0], S_old[1], h)
        F[1] = S2(y1, y2, S_old[0], S_old[1], h)

        S_new = S_old - aplha * (np.matmul(inv(Jnew), F))

        error = np.max(np.abs(S_new - S_old))

        S_old = S_new

        iter = iter + 1

    return [S_new[0], S_new[1]]


def implicit_euler(inty1, inty2, tspan, dt):
    t = np.arange(0, tspan, dt)
    y1 = np.zeros(len(t))
    y2 = np.zeros(len(t))

    y1[0] = inty1
    y2[0] = inty2

    y1_guess = 1
    y2_guess = 1

    for i in range(1, len(t)):
        y1[i], y2[i] = NewtonRaphson(y1[i - 1], y2[i - 1], y1_guess, y2_guess, dt)

        y1_guess = y1[i]
        y2_guess = y2[i]

    return [t, y1, y2]


t, y1, y2 = implicit_euler(2, -4, 10, 0.5)

plt.plot(t, y1, 'b')
plt.plot(t, y2, 'g')
plt.legend(["graph of R", "graph of J"])
plt.show()