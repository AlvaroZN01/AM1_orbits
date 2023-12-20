from numpy import arange, array
import matplotlib.pyplot as plt
from ODEs.Cauchy_problem import Cauchy
from ODEs.Temporal_schemes import Euler, Inverse_Euler, RK4, Crank_Nicolson

# Funciona  integrar
def F_Kepler(t, U):
    
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    mr = (x**2 + y**2)**1.5
    return array([vx, vy, -x/mr, -y/mr])

# Definicion de los parametros de integracion
dt = 0.001
N = 10000
t0 = 0
t = arange(t0, N * dt, dt)
U0 = [1, 0, 0, 1]

# Resolucion del problema de Cauchy
U = Cauchy(t, Inverse_Euler, F_Kepler, U0)

# Representacion de resultados
plt.axis('equal')
plt.plot(U[0,:],U[1,:])
plt.show()

