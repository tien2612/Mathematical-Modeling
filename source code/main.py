import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.colors
import math

# function that returns dz/dt
t = np.arange(0,100*np.pi, 1)
def model(z, t):
    R = pow(np.e, 3*t) * ( -np.cos(t) - np.sin(t) )
    J = pow(np.e, 3*t) * ( 3 * np.cos(t) +  2*np.sin(t) )
    Rdot = 2 * R + 2 * J
    Jdot = -1 * R + 4*J
    dzdt = [Rdot, Jdot]
    return dzdt


# initial condition
z0 = [-1, 3]

# time points
t = np.linspace(0, 6, 1000)
# solve ODE
z = odeint(model, z0, t)

# plot results
plt.plot(t, z[:,0],'c-',label=r"Romeo's")
plt.plot(t, z[:,1], color = 'orange', label=r"Juliet's")
plt.title("LOVE BETWEEN A EAGER BEAGER AND A NARCISSISTIC NERD", fontsize = 12)
plt.ylabel('Love for the other')
plt.xlabel('Time')
plt.legend(loc='best')
plt.grid()
plt.show()