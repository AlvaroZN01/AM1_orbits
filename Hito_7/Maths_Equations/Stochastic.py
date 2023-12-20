
#=================================================================================
#								     STOCHASTIC
#=================================================================================

from numpy import sqrt, linspace, cumsum, random, concatenate, zeros
import matplotlib.pyplot as plt
from Plotting.Graphics import Plot_2D


def Proceso_Wiener(T, N):
    dt = T / N
    t = linspace(0.0, T, N+1)
    dW = sqrt(dt) * random.randn(N)
    W =cumsum(dW)

    Plot_2D(t, concatenate(([0], W)),Label_X='Tiempo',Label_Y = 'Valor del proceso')



def Proceso_OrnsteinUhlenbeck(theta, sigma, T, N):
    dt = T / N
    t = linspace(0.0, T, N+1)
    dW = sqrt(dt) * random.randn(N)
    
    X = zeros(N+1)
    X[0] = 0.0  # Condición inicial
    
    for i in range(N):
        X[i+1] = X[i] - theta * - X[i] * dt + sigma * dW[i]
    
    Plot_2D(t,X,Label_X = '$t$',Label_Y = '$X_t$')
