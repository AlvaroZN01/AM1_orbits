from numpy import arange, array, log10, linalg, zeros, vstack, ones, linspace, transpose, reshape
from numpy.linalg import norm, lstsq
import matplotlib.pyplot as plt
from ODEs.Cauchy_problem import Cauchy
from ODEs.Temporal_schemes import Euler, Inverse_Euler, RK4, Crank_Nicolson
from ODEs.Temporal_error import Error_Cauchy_Problem, Richardson_extrapolation
from ODEs.Stability import Stability_region
from Equations.Equations import F_Kepler, Oscillator, N_body

# Definicion de una funcion adicional, ya que la funcion de Cauchy solo da como inputs t y U, y la funcion
# necesita tambien Nb y Nc.
def F_NBody(t, U):
    return N_body(t, U, Nb, Nc)

# Funcion para definir y guardar las condiciones iniciales en el vector U0
def Init_cond(Nb, Nc):
    U0 = zeros(2*Nc*Nb)
    U1 = reshape(U0, (Nb, Nc, 2))  
    r0 = reshape(U1[:, :, 0], (Nb, Nc))
    v0 = reshape(U1[:, :, 1], (Nb, Nc))

    r0[0,:] = [1, 0, 0]
    v0[0,:] = [0, 0.5, 0]

    v0[1,:] = [0, -1, 0]
    r0[1,:] = [-0.5, 0, 0]

    r0[2,:] = [0, 0, 1]
    v0[2,:] = [0, 0, 0.5]

    r0[3,:] = [0, 1, 0]
    v0[3,:] = [0, 0, -0.5]

    r0[4,:] = [-1, 0, 0] 
    v0[4,:] = [0, -0.5, 0]
    return U0 

# Definicion de los parametros de integracion
N = 10000
t0 = 0
tf = 5
t = linspace(t0, tf, N+1)

# Definicion del numero de cuerpos y del numero de coordenadas por cuerpo
Nb = 5
Nc = 3

# Resolucion del prolema de Cauchy
U0 = Init_cond(Nb, Nc)
U = Cauchy(t, RK4, F_NBody, U0)

# Representacion de resultados (en 3 dimensiones)
Us = reshape(U, (Nb, Nc, 2, N+1))
r = reshape(Us[:, :, 0, :], (Nb, Nc, N+1))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(Nb):
    ax.plot3D(r[i, 0, :], r[i, 1, :], r[i, 2, :], label=f'Cuerpo {i+1}')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()