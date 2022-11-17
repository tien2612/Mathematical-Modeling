import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.colors
import math

# function that returns dz/dt
t = np.arange(0, 10)
def model(z, t):
    #example 1
    a = 1
    b = 6
    c = 1
    d = -4
    c1 = 1
    c2 = -1
    R = 6 * c1 * pow(np.e, 2*t) + 3 * c2 * pow(math.e, -3 * t)
    J = c1 * pow(np.e, 2*t) + 2 * c2 * pow(math.e, -3 * t)
    Rdot = a * R + b * J
    Jdot = c * R + d * J
    dzdt = [Rdot, Jdot]
    return dzdt

# initial condition
z0 = [3, -1]

# time points
t = np.linspace(0, 10, 100)
# solve ODE
z = odeint(model, z0, t)

# plot results
plt.plot(t, z[:,0],'c-',label=r"Romeo's")
plt.plot(t, z[:,1], color = 'orange', label=r"Juliet's")
plt.title("LOVE BETWEEN A EAGER BEAGER AND A CAUTIONS LOVER", fontsize = 12)
plt.ylabel('Love for the other')
plt.legend(loc='best')
plt.grid()
plt.show()