
#=================================================================================
#								  TEMPORAL ERROR
#=================================================================================


from Maths_ODEs.Cauchy_Problem import Cauchy_Problem
from Maths_ODEs.Temporal_Schemes import Explicit_Euler, Inverse_Euler, Crank_Nicolson, RungeKutta_4
from Physics.Kepler_Equation import Kepler_Equation

from numpy import array, zeros, log10, ones, vstack, linspace
from numpy.linalg import norm, lstsq
import matplotlib.pyplot as plt


def Error_Cauchy_Problem( t, F, Scheme, orden, U_0 ):  
# EXTRAPOLACIÓN DE RICHARDSON (utilizar dos mallas)
                  
    N = len(t)-1 
    Nv = len(U_0) 
    t1 = t
    t2 = zeros(2*N+1)
    Error = zeros((N+1, Nv))
       
    # Duplicar los puntos en el dominio del tiempo
    for i in range(N):  
        t2[2*i]   = t1[i] 
        t2[2*i+1] = ( t1[i] + t1[i+1] )/2
    t2[2*N] = t1[N]
      
    # Resolver el problema de Cauchy para dos conjuntos de tiempo
    U1 =   Cauchy_Problem( F, t1, U_0, Scheme) 
    U2 =   Cauchy_Problem( F, t2, U_0, Scheme)    
       
    # Calcular el error entre las soluciones
    for i in range(N+1):  
        Error[i,:] = ( U2[2*i, :]- U1[i, :] )/( 1 - 1./2**orden ) 
        
    # Corregir la solución usando el error calculado
    Sol = U1 + Error 
       
    return Error, Sol




def Temporal_Convergence_Rate(t, F, U_0, Scheme, m):

    log_E = zeros(m)        # Logaritmo de errores
    log_N = zeros(m)        # Logaritmos del tamaño del intervalo temporal

    # Resolver el problema de Cauchy inicialmente
    N = len(t) - 1
    t1 = t
    U1 = Cauchy_Problem(F, t1, U_0, Scheme)  

    for i in range(m):
        N = 2 * N                       # Se duplica el número de puntos
        t2 = array(zeros(N + 1))        # Crear un nuevo vector de tiempo con el nuevo N
        t2[0:N + 1:2] = t1              # Asignar los puntos originales al nuevo vector de tiempo
        t2[1:N:2] = (t1[1:int(N / 2) + 1] + t1[0:int(N / 2)]) / 2  # Agregar puntos intermedios al nuevo vector de tiempo

        # Resolver el problema de Cauchy con el nuevo vector de tiempo
        U2 = Cauchy_Problem(F, t2, U_0, Scheme)  

        # Calcular el error entre las soluciones en puntos específicos del tiempo
        error = norm(U2[N, :] - U1[int(N / 2), :])

        # Almacenar Logaritmos de error y de N
        log_E[i] = log10(error)
        log_N[i] = log10(N)
        t1 = t2         # Actualizar los tiempos para la siguiente iteración
        U1 = U2         # Actualizar la solución para la siguiente iteración


    # Determinar el número de iteraciones donde los errores son significativos
    for j in range(m):
        if abs(log_E[j]) > 12:  # Si el error es mayor a 10^-12
            break

    j = min(j, m - 1)       # Tomar el valor mínimo entre j y m-1
    x = log_N[0:j + 1]      # Subconjunto de logaritmos del tamaño del intervalo temporal
    y = log_E[0:j + 1]      # Subconjunto de logaritmos de errores

    A = vstack([x, ones(len(x))]).T     # Preparar los datos para la recta de regresión
    m, c = lstsq(A, y, rcond=None)[0]   # Regresion
    orden = abs(m)                      # Valor absoluto de la pendiente

    # Corregir los errores basados en la tasa de convergencia calculada (valor de la pendiente en abs)
    log_E = log_E - log10(1 - 1. / 2 ** orden)

    return orden, log_E, log_N
