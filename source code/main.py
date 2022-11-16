import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.colors
import math

# function that returns dz/dt
t = np.arange(0, 11)
R = -4 * pow(math.e, -t)
J = 4 * pow(math.e, t)
def model(z,t):
    R = 4/3 * pow(math.e, 3 * t) - 1/3
    J = 2/3 * pow(math.e, 3 * t) + 1/3
    Rdot = 2 * R + 2 * J
    Jdot = R + J
    dzdt = [Rdot, Jdot]
    return dzdt

# initial condition
z0 = [1, 1]

# time points
t = np.linspace(0, 11)
# solve ODE
z = odeint(model,z0,t)

# plot results
plt.plot(t, z[:,0],'c-',label=r'Romeo')
plt.plot(t, z[:,1], color = 'orange', label=r'Juliet')
plt.ylabel('Love for the other')
plt.xlabel('Time')
plt.legend(loc='best')
plt.grid()
plt.show()