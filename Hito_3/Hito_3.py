from numpy import arange, array, zeros, linalg, log10
import matplotlib.pyplot as plt
from scipy import optimize


# Temporal schemes to integrate a Cauchy problem
def Euler(U0, t0, tf, f):
    return U0 + (tf - t0) * f(t0, U0)

def Crank_Nicolson(U0, t0, tf, f):
    def Residual(x):
        return x - U0 - (tf - t0)/2 * (f(t0, U0) + f(tf, x))
    return optimize.newton(func = Residual, x0 = U0)

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
def Cauchy(t, temporal_scheme, f, U0):
    U = array(zeros((len(U0),len(t))))
    U[:,0] = U0
    for ii in range(0, N - 1):
        U[:,ii+1] = temporal_scheme(U[:,ii], t[ii], t[ii+1], f)
    return U

# Error evaluation using Richardson extrapolation 
def error_eval(tn, temporal_scheme, f, U0):
    t2n = array(zeros((len(t)*2)))
    t2n[0] = tn[0]
    for ii in range(1, len(t)):
        t2n[ii*2] = tn[ii]
        t2n[ii*2-1] = (tn[ii] + tn[ii-1]) / 2
    U = Cauchy(tn, temporal_scheme, f, U0)
    V = Cauchy(t2n, temporal_scheme, f, U0)
    log_norm_E_n = array(zeros((len(t))))
    for ii in range(0, len(t)):
        log_norm_E_n[ii] = log10(linalg.norm(V[:,ii*2] - U[:,ii]))
    n_vec = arange(0, len(t), 1)
    n_vec = log10(n_vec)
    plt.plot(n_vec, log_norm_E_n) # Según las gráficas de clase, debería salir con pendiente negativa.
    plt.show()
    return n_vec


# Function to be integrated
def F_Kepler(t, U):
    
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    mr = (x**2 + y**2)**1.5
    return array([vx, vy, -x/mr, -y/mr])

# Integration parameters definition
dt = 0.001
N = 10000
t0 = 0
t = arange(t0, N * dt, dt)
U0 = [1, 0, 0, 1]

# Cauchy problem solver
#U = Cauchy(t, Crank_Nicolson, F_Kepler, U0)

# Richardson extrapolation
n = error_eval(t, Euler, F_Kepler, U0)

# Plot results
# plt.axis('equal')
# plt.plot(U[0,:],U[1,:])
# plt.show()
