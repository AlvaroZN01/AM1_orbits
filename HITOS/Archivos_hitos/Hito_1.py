from numpy import array, zeros
import matplotlib.pyplot as plt
from scipy import optimize


def F_Kepler(U): # Definición de la función a integrar (ecuación de las orbitas de Kepler)
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    mr = (x**2 + y**2)**1.5
    return array([vx, vy, -x/mr, -y/mr])

def G(x): # Definición del esquema numérico para integrar la ecuación (Crank-Nicolson)
    return x - A - dt/2 * (F_Kepler(A) + F_Kepler(x))


# Definición de los parámetros temporales
N = 10000
dt = 0.001

# ----- Método de Euler -----
# Definición del vector solución y de las condiciones iniciales
U = array(zeros((4,N))) 
U[:,0] = [1, 0, 0, 1]

for ii in range(0,N-1):
    F = F_Kepler(U[:,ii])
    U[:,ii+1] = U[:,ii] + dt * F # Integración mediante el método de Euler

# Representación de resultados
plt.axis('equal')
plt.plot(U[0,:],U[1,:])
plt.show()

# ----- Método de Crank-Nicolson -----
# Definición del vector solución y de las condiciones iniciales
U = array(zeros((4,N)))
U[:,0] = [1, 0, 0, 1]

for ii in range(0,N-1):
    A = U[:,ii]
    U[:,ii+1] = optimize.newton(func = G, x0 = A) # Integración mediante el método de Crank-Nicolson (implicito)

# Representación de resultados
plt.axis('equal')
plt.plot(U[0,:],U[1,:])
plt.show()
    
# ----- Método de Runge Kutta 4 -----
# Definición del vector solución y de las condiciones iniciales
U = array(zeros((4,N)))
U[:,0] = [1, 0, 0, 1]

for ii in range(0,N-1):
    k1 = F_Kepler(U[:,ii])
    k2 = F_Kepler(U[:,ii] + k1*dt/2)
    k3 = F_Kepler(U[:,ii] + k2*dt/2)
    k4 = F_Kepler(U[:,ii] + k3*dt)
    U[:,ii+1] = U[:,ii] + dt/6 * (k1 + 2*k2 + 2*k3 + k4) # Integración mediante el método de Runge Kutta 4

# Representación de resultados
plt.axis('equal')
plt.plot(U[0,:],U[1,:])
plt.show()

