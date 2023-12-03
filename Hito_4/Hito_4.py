from numpy import arange, array, log10, linalg, zeros, vstack, ones, linspace, transpose
from numpy.linalg import norm, lstsq
import matplotlib.pyplot as plt
from ODEs.Cauchy_problem import Cauchy
from ODEs.Temporal_schemes import Euler, Inverse_Euler, RK4, Crank_Nicolson
from ODEs.Temporal_error import Error_Cauchy_Problem, Convergence_rate
from ODEs.Stability import Stability_region

# Function to be integrated
def F_Kepler(t, U):
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    mr = (x**2 + y**2)**1.5
    return array([vx, vy, -x/mr, -y/mr])

def Oscillator(t, U):
    return array([U[1], -U[0]])

# Integration parameters definition
dt = 0.1
N = 1001
t0 = 0
t = arange(t0, N * dt, dt)
U0 = [1, 0]

# Cauchy problem solver
U = Cauchy(t, Euler, Oscillator, U0)
plt.plot(t, U[1,:])
plt.show()

U = Cauchy(t, Inverse_Euler, Oscillator, U0)
plt.plot(t, U[1,:])
plt.show()

U = Cauchy(t, RK4, Oscillator, U0)
plt.plot(t, U[1,:])
plt.show()

U = Cauchy(t, Crank_Nicolson, Oscillator, U0)
plt.plot(t, U[1,:])
plt.show()

# Stability regions
rho, x, y = Stability_region(Euler, 100, -4, 4, -4, 4)
plt.contour(x, y, transpose(rho), linspace(0,1,11))
plt.grid()
plt.show()

rho, x, y = Stability_region(Inverse_Euler, 100, -4, 4, -4, 4)
plt.contour(x, y, transpose(rho), linspace(0,1,11))
plt.grid()
plt.show()

rho, x, y = Stability_region(RK4, 100, -4, 4, -4, 4)
plt.contour(x, y, transpose(rho), linspace(0,1,11))
plt.grid()
plt.show()

rho, x, y = Stability_region(Crank_Nicolson, 100, -4, 4, -4, 4)
plt.contour(x, y, transpose(rho), linspace(0,1,11))
plt.grid()
plt.show()




