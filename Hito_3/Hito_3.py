from numpy import arange, array, log10, linalg, zeros
import matplotlib.pyplot as plt
from ODEs.Cauchy_problem import Cauchy
from ODEs.Temporal_schemes import Euler, Inverse_Euler, RK4, Crank_Nicolson

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
    plt.plot(n_vec, log_norm_E_n) # Segun las graficas de clase, deberia salir con pendiente negativa.
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
