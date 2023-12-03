from numpy import array, log10, zeros, vstack, ones
from numpy.linalg import norm, lstsq
import matplotlib.pyplot as plt
from ODEs.Cauchy_problem import Cauchy

# Error evaluation using Richardson extrapolation 
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

# Temporal convergence rate using Richardson extrapolation 
def Convergence_rate(t1, temporal_scheme, f, U0, m):
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