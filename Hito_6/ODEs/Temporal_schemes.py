from scipy.optimize import newton
from numpy import zeros, dot
from numpy.linalg import norm

# Esquema temporal Euler
def Euler(U0, t0, tf, f):
    return U0 + (tf - t0) * f(t0, U0)


# Esquema temporal Crank Nicolson
def Crank_Nicolson(U0, t0, tf, f):
    def Residual(x):
        return x - U0 - (tf - t0)/2 * (f(t0, U0) + f(tf, x))
    return newton(func = Residual, x0 = U0, maxiter = 250)


# Esquema temporal Runge Kutta 4
def RK4(U0, t0, tf, f):
    dt = tf - t0
    k1 = f(t0, U0)
    k2 = f(t0 + dt/2, U0 + k1*dt/2)
    k3 = f(t0 + dt/2, U0 + k2*dt/2)
    k4 = f(t0 + dt, U0 + k3*dt)
    return U0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


# Esquema temporal Euler inverso
def Inverse_Euler(U0, t0, tf, f):
    def Residual(x):
        return x - U0 - (t0 - tf) * f(tf, x)
    return newton(func = Residual, x0 = U0, maxiter = 250)


# Esquema temporal Runge Kutta embebido
def Embedded_RK(U0, t0, tf, f, q, Tolerance): 
    dt = tf - t0

    a, b, bs, c = obtener_array_Butcher(q)
    k = RK_k_calculation(f, U0, t0, dt, a, c)

    # Error = dot(b-bs, k)
    Error = dt * dot(bs-b, k)
    dt_min = min(dt, dt * (Tolerance / norm(Error))**(1/q))
    N = int(dt/dt_min) + 1
    h = dt / N
    Uh = U0[:]

    for i in range(0, N):
        k = RK_k_calculation(f, Uh, t0 + h*i, h, a, c)
        Uh += h * dot(b, k)

    return Uh

# Calculo de los valores de las k
def RK_k_calculation(f, U0, t0, dt, a, c):
     k = zeros((len(c), len(U0)))
     for i in range(len(c)):
        Up = U0 + dt * dot(a[i, :], k)
        k[i,:] = f(t0 + c[i]*dt, Up)
     return k

# Funcion para obtener la matriz de Butcher para diferentes órdenes de Runge Kutta
def obtener_array_Butcher(q): 
    N_stages = {2:2, 3:4, 8:13}

    N =  N_stages[q]
    a = zeros((N, N))
    b = zeros((N))
    bs = zeros((N))
    c = zeros((N)) 
    
    if q==2:
     a[0,:] = [0, 0]
     a[1,:] = [1, 0]

     b[:] = [1/2, 1/2]

     bs[:] = [1, 0]

     c[:]  = [0, 1]

    elif q==3:
      a[0,:] = [0, 0, 0, 0]
      a[1,:] = [1/2, 0, 0, 0]
      a[2,:] = [0, 3/4, 0, 0]
      a[3,:] = [2/9, 1/3, 4/9, 0]

      b[:]  = [2/9, 1/3, 4/9, 0]

      bs[:] = [7/24, 1/4, 1/3, 1/8]

      c[:] = [0, 1/2, 3/4, 1]
   
    elif q==8:
       a[0,:] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
       a[1,:] = [2/27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
       a[2,:] = [1/36, 1/12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
       a[3,:] = [1/24, 0, 1/8 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
       a[4,:] = [5/12, 0, -25/16, 25/16, 0, 0, 0, 0, 0, 0, 0, 0, 0]
       a[5,:] = [1/20, 0, 0, 1/4, 1/5, 0, 0, 0, 0, 0, 0, 0, 0] 
       a[6,:] = [-25/108, 0, 0, 125/108, -65/27, 125/54, 0, 0, 0, 0, 0, 0, 0] 
       a[7,:] = [31/300, 0, 0, 0, 61/225, -2/9, 13/900, 0, 0, 0, 0, 0, 0] 
       a[8,:] = [2, 0, 0, -53/6, 704/45, -107/9, 67/90, 3, 0, 0, 0, 0, 0] 
       a[9,:] = [-91/108, 0, 0, 23/108, -976/135, 311/54, -19/60, 17/6, -1/12, 0, 0, 0, 0] 
       a[10,:] = [2383/4100, 0, 0, -341/164, 4496/1025, -301/82, 2133/4100, 45/82, 45/164, 18/41, 0, 0, 0] 
       a[11,:] = [3/205, 0, 0, 0, 0, -6/41, -3/205, -3/41, 3/41, 6/41, 0, 0, 0]
       a[12,:] = [-1777/4100, 0, 0, -341/164, 4496/1025, -289/82, 2193/4100, 51/82, 33/164, 19/41, 0, 1, 0]
      
       b[:]  = [41/840, 0, 0, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280, 41/840, 0, 0]

       bs[:] = [0, 0, 0, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280, 0, 41/840, 41/840]   

       c[:] = [0, 2/27, 1/9, 1/6, 5/12, 1/2, 5/6, 1/6, 2/3 , 1/3, 1, 0, 1]  
     
    else:
        print("Butcher array  not avialale for order =", q)
        exit()

    return a, b, bs, c 