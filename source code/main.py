import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.lines as mlines
import math
from scipy.integrate import odeint
from matplotlib import pyplot
from scipy import integrate
from random import uniform
from matplotlib.lines import Line2D

# defind system of equations for plot nullcline
# example 1
# initital condition
# z0 = [-1, 3]
# a = 2
# b = 2
# c = -1
# d = 4
# R_1 = np.linspace(-10, 10, 100)
# R_2 = np.linspace(-10, 10, 100)
# J_1 = -R_1
# J_2 = R_2/4
# example 2
# initial condition
z0 = [2, -2]
a = 5
b = 3
c = -4
d = 3
R_1 = np.linspace(-10, 10, 100)
R_2 = np.linspace(-10, 10, 100)
J_1 = -5*R_1/3
J_2 = 4*R_2/3

# return dR/dt and dJ/dt
def f(Y, t):
    R, J = Y
    return [a*R + b*J, c*R + d*J]

# function that returns dz/dt
def model(z, t):
    #example 1
    # c1 = -1/2
    # c2 = 7/2
    # R = pow(np.e, 3*t) * (c1 * 2 * np.cos(t) + c2 * 2 * np.sin(t))
    # J = pow(np.e, 3*t) * (c1 * (np.cos(t) - np.sin(t) )+ c2 * ( np.sin(t) + np.cos(t)) )

    #example 2
    c1 = -2/3
    c2 = 4*math.sqrt(11)/3
    R = pow(np.e, 4*t) * (3*c1*np.cos(math.sqrt(11)*t) + 3*c2*np.sin(math.sqrt(11)*t))
    J = pow(np.e, 4*t) * (c1*(-np.cos(math.sqrt(11)*t) - math.sqrt(11)*np.sin(math.sqrt(11)*t))
                          + c2*(-np.sin(math.sqrt(11)*t) + math.sqrt(11)*np.cos(math.sqrt(11)*t)) )

    Rdot = a * R + b * J
    Jdot = c * R + d * J
    dzdt = [Rdot, Jdot]
    return dzdt

# time points
t = np.linspace(0, 2, 500)
#
# solve ODE
z0 = [-2, 2]
z = odeint(model, z0, t)
# create plot
plt.plot(t, z[:,0],'c-',label=r"Romeo's")
plt.plot(t, z[:,1], color = 'orange', label=r"Juliet's")
plt.title("LOVE BETWEEN A EAGER BEAGER AND A CAUTIONS LOVER", fontsize = 12)
plt.ylabel('Love for the other')
plt.legend(loc='best')
plt.grid()

# phase portrait
# mesh grid for each point R, J then store into u, v
R = np.linspace(-5.0, 5.0, 10)
J = np.linspace(-5.0, 5.0, 10)
RR, JJ = np.meshgrid(R, J)
u, v = np.zeros(RR.shape), np.zeros(JJ.shape)
# plot vector field
NI, NJ = RR.shape
t = 0
for i in range(NI):
    for j in range(NJ):
        x = RR[i, j]
        y = JJ[i, j]
        yprime = f([x, y], t)
        u[i, j] = yprime[0]
        v[i, j] = yprime[1]

fig, ax = plt.subplots(1, 1)
M = np.hypot(u, v)

# Fix size for all arrows equal of length
r = (u**2+v**2)**0.5
np.seterr(divide='ignore', invalid='ignore')
q = ax.quiver(RR, JJ, u/r, v/r, M, units='x', pivot='tip', width=0.07)

ax.set(xlim=(-4.7, 4.7), ylim=(-4.7, 4.7))
ax.set_aspect(aspect=1)

ax.set_title("LOVE BETWEEN A EAGER BEAGER AND A NARCISSISTIC NERD", fontsize = 10)
ax.set_ylabel("Juliet's love for Romeo", fontsize=12)
ax.set_xlabel("Romeo's love for Juliet", fontsize=12)
plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4]) # set ticks

# Plot trajectories by looping through the possible values
# generate random initial value and draw 5 trajactories
random_init = [(uniform(-4.7, 4.7), uniform(-4.7, 4.7)) for i in range(5)]
#random_init = [(-5, 5), (5, 0), (-3, 5), (0, 4), (-3, 2)]
t = np.linspace(-5, 5, 1000)
print(random_init)
random_index = 0
for v in random_init:
    sol = odeint(f, v, t)
    if random_index % 2 == 0:
        traject1 = plt.plot(sol[:, 0], sol[:, 1], '-', c = 'gray', label = r'Trajectory')
    else:
        traject2 = plt.plot(sol[:, 0], sol[:, 1], '-', c ='r' ,label = r'Trajectory')
    random_index += 1

# Plot nullcline and fixed point
nullcline1 = ax.plot(R_1, J_1, '--', dashes=(3, 1), linewidth= 0.9 ,c = 'b',label = r'Nullcline 1')
nullcline2 = ax.plot(R_2, J_2, '--', dashes=(3, 1), linewidth= 0.9 ,c = 'magenta',label = r'Nullcline 2')
# Create fixed point
fixed_point = plt.plot(0, 0, 'go', label = 'Fixed point')
plt.setp(fixed_point, markersize=8)
plt.setp(fixed_point, markerfacecolor='white', color = 'gray')
# Show legend
# Add arrow chacracter to legend for vector field
arrow = u'$\u279E$'
legend_elements = [Line2D([0], [0], marker = arrow, color='white', label='Vector field',
                          markerfacecolor='#0BA3A7', markersize=20)]

all_legends = traject2 + traject1 + nullcline1 + nullcline2 + fixed_point
labs = [l.get_label() for l in all_legends]
ax.legend(handles=all_legends + legend_elements, loc='upper left')
# Show all plots
plt.show()