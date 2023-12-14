from numpy import arange, array, log10, zeros, vstack, ones
from numpy.linalg import norm, lstsq
import matplotlib.pyplot as plt
from ODEs.Cauchy_problem import Cauchy
from ODEs.Temporal_schemes import Euler, Inverse_Euler, RK4, Crank_Nicolson

# Evaluacion del error empleando la extrapolacion de Richardson 
def Error_Cauchy_Problem( t1, temporal_scheme, U0, f, order):  
    N = len(t1)-1
    t1 = t1
    t2 = zeros(2*N+1)
    error = zeros((len(U0), N+1))
       
    for i in range(N):
        t2[2*i] = t1[i] 
        t2[2*i+1] = (t1[i] + t1[i+1]) / 2

    t2[2*N] = t1[N]
      
    U1 = Cauchy(t1, temporal_scheme, f, U0) 
    U2 = Cauchy(t2, temporal_scheme, f, U0) 
       
    for i in range(N+1):  
        error[:,i] = (U2[:,2*i] - U1[:,i]) / (1 - 1./2**order) 
       
    return error, U1 + error 

# Ratio de convergencia temporal empleando la extrapolacion de Richardson 
def Richardson_extrapolation(t1, temporal_scheme, f, U0, m):
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

# Funcion a integrar (ecuacion de las orbitas de Kepler)
def F_Kepler(t, U):
    
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    mr = (x**2 + y**2)**1.5
    return array([vx, vy, -x/mr, -y/mr])

# Definicion de los parametros de integracion
dt = 0.1
N = 1001
t0 = 0
t = arange(t0, N * dt, dt)
U0 = [1, 0, 0, 1]

# Resolucion del problema de Cauchy
U = Cauchy(t, RK4, F_Kepler, U0)
plt.plot(U[0,:],U[1,:])
plt.show()

# Estimacion del error
[error_U, U_wne] = Error_Cauchy_Problem(t, RK4, U0, F_Kepler, 1)
plt.plot(error_U[0,:],error_U[1,:])
plt.show()
plt.plot(U_wne[0,:],U_wne[1,:])
plt.show()

# Ratio de convergencia temporal
# order = Richardson_extrapolation(t, Euler, F_Kepler, U0, 10)
# print('The order of the Euler scheme is', order)
# order = Richardson_extrapolation(t, Inverse_Euler, F_Kepler, U0, 10)
# print('The order of the Inverse Euler scheme is', order)
# order = Richardson_extrapolation(t, RK4, F_Kepler, U0, 10)
# print('The order of the Runge Kutta 4 scheme is', order)
# order = Richardson_extrapolation(t, Crank_Nicolson, F_Kepler, U0, 10)
# print('The order of the Crank Nicolson scheme is', order)
