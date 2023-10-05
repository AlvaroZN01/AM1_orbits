from numpy import array, zeros, linalg
import matplotlib.pyplot as plt
from scipy import optimize


# Temporal schemes to integrate a Cauchy problem
def Euler(U0, t0, tf, f):
    return U0 + (tf - t0) * f(t0, U0)

def Crank_Nicolson(U0, t0, tf, f):
    def Residual(x):
        return x - U0 - (tf - t0)/2 * (f(t0, U0) + f(tf, x))
    return optimize.newton(func = Residual, x0 = U0)

def F_CK(x, U0, t0, tf, f):
    return x - U0 - (tf - 0)/2 * (f(t0, U0) + f(tf, x))

def RK4(U0, t0, tf, f):
    dt = tf - t0
    k1 = f(t0, U0)
    k2 = f(t0 + dt/2, U0 + k1*dt/2)
    k3 = f(t0 + dt/2, U0 + k2*dt/2)
    k4 = f(t0 + dt, U0 + k3*dt)
    return U0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def Inverse_Euler(U0, t0, tf, f):
    def Residual(x):
        return x - U0 - (t0 - tf) * f(tf, x)
    return optimize.newton(func = Residual, x0 = U0)

# Cauchy problem solver
# def Cauchy(t, temporal_scheme, f, U0)
def Cauchy(Solver, f, U0, t0, dt, N):
    # U = array (zeros((len(U0),len(t))))
    U = array(zeros((len(U0),N)))
    U[:,0] = U0
    # Esto fuera
    t = t0
    for ii in range(0, N - 1):
        # U[:,ii+1] = temporal_scheme(U[:,ii], t[ii], t[ii+1], f)
        U[:,ii+1] = Solver(U[:,ii], t, t + dt, f)
        # Esto fuera
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

