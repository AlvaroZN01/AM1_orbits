from numpy import arange, array, log10, zeros, vstack, ones, linspace, transpose, reshape, sqrt, round, pi, cos, sin, meshgrid
from numpy.linalg import norm, lstsq
from scipy.optimize import fsolve, root, newton
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ODEs.Cauchy_problem import Cauchy
from ODEs.Temporal_schemes import Euler, Inverse_Euler, RK4, Crank_Nicolson, Embedded_RK
from ODEs.Temporal_error import Error_Cauchy_Problem, Richardson_extrapolation
from ODEs.Stability import Stability_region
from Equations.Equations import F_Kepler, Oscillator, N_body, N3_body_restricted, Stability_Lagrange

# Definicion de una funcion adicional para pasar los valores definidos de q y tol a la funcion Embedded_RK
def RK_emb(U0, t0, tf, f):
    return Embedded_RK(U0, t0, tf, f, q, tol)

# Definicion de una funcion adicional para pasar los valores definidos de m1, m2, pos1 y pos2 a la funcion N3_body_restricted
def CR3BP(t, U):
    return N3_body_restricted(t, U, m1, m2, r12)

# Funcion para obtener la posicion de los planetas en eje x, en ejes centrados en el baricentro del sistema compuesto por m1 y m2
def obtener_pos_planetas(m1, m2, r12):
    pi1 = m1 / (m1 + m2)
    pi2 = m2 / (m1 + m2)
    pos1 = -r12*pi2
    pos2 = r12*pi1
    return(pos1, pos2)

def CR3BP_Lagrange(x):
    U = array([x[0], x[1], 0, 0])
    Sol = N3_body_restricted(0, U, m1, m2, r12)
    return array([Sol[0], Sol[1]])

# Definicion del orden y tolerancia del RK embebido
q = 8
tol = 1e-12

# Definicion de los parametros del problema de los 3 cuerpos circular restringido. Se incluyen los datos Sol-Tierra y Tierra-Luna
m1 = 3.955e30 # Masa del Sol (kg)
m2 = 5.972e24 # Masa de la Tierra (kg)
r12 = 149597870 # Distancia Sol-Tierra [km]
R_1 = 696340 # Radio del Sol, para la representacion [km]
R_2 = 6378 # Radio de la Tierra, para la representacion [km]

# m1 = 5.972e24 # Masa de la Tierra (kg)
# m2 = 7.349e22 # Masa de la Luna (kg)
# r12 = 384403 # Distancia Tierra-Luna [km]
# R_1 = 6378 # Radio de la Tierra, para la representacion [km]
# R_2 = 1737 # Radio de la Luna, para la representacion [km]

# m1 = 3.955e25 # Masa del Sol (kg)
# m2 = 5.972e24 # Masa de la Tierra (kg)
# r12 = 1495978 # Distancia Sol-Tierra [km]
# R_1 = 69634 # Radio del Sol, para la representacion [km]
# R_2 = 6378 # Radio de la Tierra, para la representacion [km]

# Definicion de los parametros de integracion
N = 10000
t0 = 0
tf = 10000
t = linspace(t0, tf, N+1)

# Resolucion del problema de Cauchy
pos1, pos2 = obtener_pos_planetas(m1, m2, r12)
U0 = [pos2 + 6378 + 500, 0, 0, 11]
U = Cauchy(t, RK_emb, CR3BP, U0)

# Calculo de puntos criticos
initial_guess = array([[0.8*r12, 0], [1.2*r12, 0], [-1*r12, 0], [0.5*r12, 0.5*r12], [0.5*r12, -0.5*r12]])
sol_Lagrange = zeros((2,5))
for ii in range(0,5):
    sol_Lagrange[:,ii] = newton(CR3BP_Lagrange, initial_guess[ii])

for ii in range(0,5):
    for jj in range(0,2):
        if abs(sol_Lagrange[jj,ii]) < 1e-4:
            sol_Lagrange[jj,ii] = 0

# print(sol_Lagrange)
            
# Cálculo de la estabilidad de los puntos de Lagrange
x = linspace(-r12*2, r12*2, 1000)
y = linspace(-r12*2, r12*2, 1000)
Potencial = Stability_Lagrange(x, y, m1, m2, r12)

# Representacion de resultados:
fig, ax = plt.subplots()
# - Representacion de las masas (se representan a escala real, por lo que es necesario ampliar en el punto entre L1 y L2 para distinguir la masa 2 si esta es mucho mas pequeña que la 1)
m1_circle = plt.Circle((pos1, 0), R_1, color='yellow', label='M1')
m2_circle = plt.Circle((pos2, 0), R_2, color='blue', label='M2')
ax.add_patch(m1_circle)
ax.add_patch(m2_circle)
# - Representacion de las trayectorias de las masas alrededor del baricentro
theta_m = linspace(0, 2*pi, 1000)
ax.plot(abs(pos1)*cos(theta_m), abs(pos1)*sin(theta_m), color='orange', linewidth=0.25)
ax.plot(abs(pos2)*cos(theta_m), abs(pos2)*sin(theta_m), color='blue', linewidth=0.25)
# - Representacion de la trayectoria
ax.plot(U[0,:],U[1,:], label='Trayectoria')
# - Representacion de los puntos de Lagrange
cmap = plt.get_cmap('Greens')
for ii in range(0, 5):
    color = cmap((8-ii) / 8)
    ax.plot(sol_Lagrange[0, ii], sol_Lagrange[1, ii], marker='o', color=color)
    ax.text(sol_Lagrange[0, ii], sol_Lagrange[1, ii], f'L{ii+1}', color='black', fontsize=10, ha='right', va='bottom')

ax.legend()
plt.axis('equal')
plt.show()
# - Representacion del potencial gravitatorio (estabilidad de los puntos de Lagrange)
x_mesh, y_mesh = meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x_mesh, y_mesh, Potencial, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Potencial gravitatorio')
ax.set_title('Estabilidad de los puntos de Lagrange')
plt.show()