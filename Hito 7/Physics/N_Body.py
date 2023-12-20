
#=================================================================================
#								  N-BODY PROBLEM
#=================================================================================


from Maths_ODEs.Cauchy_Problem import Cauchy_Problem
from Maths_ODEs.Temporal_Schemes import RungeKutta_4

from numpy import array, zeros, reshape, shape, linspace, concatenate, split, ceil, sqrt    
from numpy.linalg import norm
from scipy.integrate import odeint, ode, solve_ivp
import matplotlib.pyplot as plt



# FUNCION F DEL PROBLEMA DE LOS N CUERPOS (lo que queda a la derecha de la ec. diferencial)
def F_NCuerpos(U, t, Nb, Nc): 
     
     # Reshape de la solucion y las derivadas a una forma tridimensional     
     Us  = reshape( U, (Nb, Nc, 2) )  
     F =  zeros(len(U))   
     dUs = reshape( F, (Nb, Nc, 2) )  
     
     # Obtener las posiciones y velocidades de la solucion
     r = reshape( Us[:, :, 0], (Nb, Nc) )
     v = reshape( Us[:, :, 1], (Nb, Nc) )
     
     # Obtener las derivadas de las posiciones (velocidades) y de las velocidades (aceleraciones)
     drdt = reshape( dUs[:, :, 0], (Nb, Nc) )
     dvdt = reshape( dUs[:, :, 1], (Nb, Nc) )
    
     # Inicializar las derivadas de velocidad en cero
     dvdt[:,:] = 0
    
     for i in range(Nb):   
       drdt[i,:] = v[i,:]    # Las derivadas de las posiciones son las velocidades actuales
       for j in range(Nb): 
         if j != i:  
           d = r[j,:] - r[i,:]  # Vector distancia entre cuerpos i y j
           dvdt[i,:] = dvdt[i,:] +  d[:] / norm(d)**3 # Derivada de velocidad
    
     return F

 


# OBTENER LAS ORBITAS DEL PROBLEMA DE LOS N CUERPOS, Y GRAFICARLAS   
def Resolver_NCuerpos():  
    
    # Definir la funcion F para el problema de N cuerpos
    def F(U, t): 
        # Llama a la funcion que calcula la F
        return F_NCuerpos(U, t, Nb, Nc)  

    N = 1000
    Nb = 4      # Numero de cuerpos
    Nc = 3      # Numero de coordenadas
    Nt = (N + 1) * 2 * Nc * Nb  # Dimension del vector de estado

    t0 = 0          # Tiempo inicial
    tf = 10         # Tiempo final
    t = linspace(t0, tf, N + 1)  # Vector de tiempo

    # CONDICIONES INICIALES
    U_0 = zeros(2 * Nc * Nb)
    U1 = reshape(U_0, (Nb, Nc, 2))
    r0 = reshape(U1[:, :, 0], (Nb, Nc))  # Posiciones iniciales
    v0 = reshape(U1[:, :, 1], (Nb, Nc))  # Velocidades iniciales

    # Definir las posiciones y velocidades iniciales para cada cuerpo (Datos r0 y v0 del ejemplo de Juan Antonio del repositorio)
    # Cuerpo 1
    r0[0, :] = [1, 0, 0]        # Posicion inicial del cuerpo 1
    v0[0, :] = [0, 0.4, 0]      # Velocidad inicial del cuerpo 1

    # Cuerpo 2
    r0[1, :] = [-1, 0, 0]       # Posicion inicial del cuerpo 2
    v0[1, :] = [0, -0.4, 0]     # Velocidad inicial del cuerpo 2

    # Cuerpo 3
    r0[2, :] = [0, 1, 0]        # Posicion inicial del cuerpo 3
    v0[2, :] = [-0.4, 0., 0.]   # Velocidad inicial del cuerpo 3

    # Cuerpo 4
    r0[3, :] = [0, -1, 0]       # Posicion inicial del cuerpo 4
    v0[3, :] = [0.4, 0., 0.]    # Velocidad inicial del cuerpo 4

    U = Cauchy_Problem(F, t, U_0, RungeKutta_4)

    Us = reshape(U, (N + 1, Nb, Nc, 2))
    r = reshape(Us[:, :, :, 0], (N + 1, Nb, Nc))    # Posiciones de cada cuerpo

    # GRAFICO
    for i in range(Nb):
        plt.plot(r[:, i, 0], r[:, i, 1])  # Trayectorias x-y de cada cuerpo
    plt.axis('equal')
    plt.grid()

    plt.show()
