from numpy import arange, array, log10, linalg, zeros, vstack, ones
from numpy.linalg import norm, lstsq
import matplotlib.pyplot as plt
from ODEs.Cauchy_problem import Cauchy
from ODEs.Temporal_schemes import Euler, Inverse_Euler, RK4, Crank_Nicolson

# Error evaluation using Richardson extrapolation 
def error_eval(t1, temporal_scheme, f, U0, m):
    log_E = zeros(m)
    log_N = zeros(m)
    N = len(t1) - 1
    U1 = Cauchy(t1, temporal_scheme, f, U0)

    for ii in range(m):
        N = 2*N
        t2 = array(zeros(N + 1))
        t2[0:N+1:2] = t1
        t2[1:N:2] = (t1[1:int(N/2)+1] + t1[0:int(N/2)]) / 2
        U2 = Cauchy(t2, temporal_scheme, f, U0)
        error = norm(U2[:,N] - U1[:,int(N/2)])
        log_E[ii] = log10(error)
        log_N[ii] = log10(N)
        t1 = t2
        U1 = U2

    plt.plot(log_N, log_E)
    plt.show()

    order, c = lstsq(vstack([log_N, ones(len(log_N))]).T, log_E, rcond=None)[0]
    order = abs(order)
    log_E = log_E - log10(1 - 1./2**order)

    return order

# Function to be integrated
def F_Kepler(t, U):
    
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    mr = (x**2 + y**2)**1.5
    return array([vx, vy, -x/mr, -y/mr])

# Integration parameters definition
dt = 1
N = 11
t0 = 0
t = arange(t0, N * dt, dt)
U0 = [1, 0, 0, 1]

# Cauchy problem solver
#U = Cauchy(t, Crank_Nicolson, F_Kepler, U0)

# Richardson extrapolation
order = error_eval(t, RK4, F_Kepler, U0, 10)
print(order)

# Plot results
# plt.axis('equal')
# plt.plot(U[0,:],U[1,:])
# plt.show()
