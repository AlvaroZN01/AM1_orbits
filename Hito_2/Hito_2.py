from numpy import array, zeros, linalg
import matplotlib.pyplot as plt
from scipy import optimize


# Temporal schemes to integrate a Cauchy problem
def Euler(f, dt, t0, U0):
    return U0 + dt * f(t0, U0)

def Crank_Nicolson(f, dt, t0, U0):
    return optimize.newton(func = F_CK, x0 = U0, fprime = None, args = (f, dt, t0, U0))

def F_CK(x, f, dt, t0, U0):
    return x - U0 - dt/2 * (f(t0, U0) + f(t0 + dt, x))

def RK4(f, dt, t0, U0):
    k1 = f(t0, U0)
    k2 = f(t0 + dt/2, U0 + k1*dt/2)
    k3 = f(t0 + dt/2, U0 + k2*dt/2)
    k4 = f(t0 + dt, U0 + k3*dt)
    return U0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def Inverse_Euler(f, dt, t0, U0):
    return optimize.newton(func = F_EI, x0 = U0, fprime = None, args = (f, dt, t0, U0))

def F_EI(x, f, dt, t0, U0):
    return x - U0 - dt * f(t0 + dt, x)

# Cauchy problem solver
def Cauchy(Solver, f, U0, t0, dt, N):
    U = array(zeros((len(U0),N)))
    U[:,0] = U0
    t = t0
    for ii in range(0, N - 1):
        U[:,ii+1] = Solver(f, dt, t, U[:,ii])
        t = t + dt
    return U



# Function to be integrated
def F_Kepler(t, U):
    
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    mr = (x**2 + y**2)**1.5
    return array([vx, vy, -x/mr, -y/mr])

# Integration parameters definition
dt = 0.001
N = 10000
U0 = [1, 0, 0, 1]
t0 = 0

# Cauchy problem solver
U = Cauchy(Inverse_Euler, F_Kepler, U0, t0, dt, N)

# Plot results
plt.axis('equal')
plt.plot(U[0,:],U[1,:])
plt.show()
