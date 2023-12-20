
#=================================================================================
#								 TEMPORAL SCHEMES
#=================================================================================

from numpy import array, zeros, linspace, matmul, size
from scipy.optimize import newton
from numpy.linalg import norm


#=================================================================================
# METODO DE EULER (EXPLICITO)
#
# El metodo de Euler (explicito) viene dado por:
#
#		U(n+1) = U(n) + delta_t * F( t_n , U(n) )

def Explicit_Euler( U, t, dt, F ):

	return U + dt * F( U, t )



#=================================================================================
# METODO DE EULER INVERSO (IMPLICITO)
#
# El metodo de Euler inverso (implicito) viene dado por:
#
#		U(n+1) = U(n) + delta_t * F( t_n , U(n+1) )

def Inverse_Euler( U, t, dt, F ):

    def G(X): 

          return X - U - dt * F( X, t )

    return newton( func = G, x0 = U ) 



#=================================================================================
# METODO CRANK-NICOLSON (IMPLICITO)
#
# El metodo de Crank-Nicolson:
#
#       U(n+1) = U(n) + (1/2) * delta_t * [ F(n+1) + F(n) ],
#
# Sera necesario resolver un sistema de ecuaciones en cada paso, al depender U(n+1) de F(n+1)
#
#		U(n+1) - U(n) - (1/2) * delta_t * F(n+1) - (1/2) * delta_t * F(n) = 0
#
# Al no ser lineal el sistema de ecuaciones, se va a emplear el metodo de Newton-Rhapson

def Crank_Nicolson( U, t, dt, F ):

    def G(X): # G = U(n+1)
         
         return  X - U - dt/2 * F( U, t ) - dt/2 * F( X, t + dt )

    return newton( func = G, x0 = U )



#=================================================================================
# METODO RUNGE-KUTTA DE 4 ETAPAS (EXPLICITO)
#
# El metodo Runge-Kutta de 4 etapas (RK4), para este problema viene dado por la siguiente ecuacion:
#
#		U(n+1) = U(n) + (1/6) * delta_t * [ k1 + 2*k2 + 2*k3 + k4 ],
#
# siendo:
#
#		k1 = F( t_n , U(t_n) )
#		k2 = F( t_n + 1/2 * delta_t , U(t_n) + 1/2 * k1 * delta_t )
#		k3 = F( t_n + 1/2 * delta_t , U(t_n) + 1/2 * k2 * delta_t )
#		k4 = F( t_n + delta_t , U(t_n) + k3 * delta_t )

def RungeKutta_4( U, t, dt, F ):
	
	k1 = F( U, t )
	k2 = F( U + dt * k1/2, t + dt/2 )
	k3 = F( U + dt * k2/2, t + dt/2 )
	k4 = F( U + dt * k3,   t + dt   )

	return  U + dt * ( k1 + 2*k2 + 2*k3 + k4 ) / 6



#===============================================================================
# METODO RUNGE-KUTTA EMBEDDED

def RungeKutta_EMB(U, t, dt, F):

    eps = 1e-8
    
    # RUNGE-KUTTA 1 y 2
    stage1 = RK_stages(1, U, t, dt, F)  
    stage2 = RK_stages(2, U, t, dt, F) 
    
    # MATRIZ DE BUTCHER
    orders, Ns, a, b, bs, c = Matriz_Butcher()
    
    # Determine the minimum step size between dt and the stepsize, which compares the error with the tolerance
    h = min(dt, StepSize(stage1 - stage2, eps, dt,  min(orders)))
    N_n = int(dt/h) + 1        # Number of steps to update solution U2
    n_dt = dt / int(N_n)           
    stage1 = U
    stage2 = U

    for i in range(N_n):
        time = t + i * n_dt
        stage1 = stage2
        stage2 = RK_stages(1, stage1, time, n_dt, F)
        
    # Final solution
    U2 = stage2
    
    ierr = 0

    return U2



def RK_stages(order, U1, t, dt, F):
    
    orders, Ns, a, b, bs, c = Matriz_Butcher()
    
    Nk = len(U1)
    # k are the intermediate values based on the butcher array coefficients
    k = zeros([Ns, Nk])
    
    k[0,:] = F(U1, t + c[0]*dt)

    if order == 1:
        for i in range(1,Ns):
            
            Up = U1
            
            for j in range(i):
                
                Up = Up + dt * a[i,j] * k[j,:]
                
            k[i,:] = F(Up, t + c[i]*dt)
            
        # Final solution
        U2 = U1 + dt*matmul(b,k)

    elif order == 2:
        for i in range(1,Ns):
            
            Up = U1
            
            for j in range(i):
                
                Up = Up + dt * a[i,j] * k[j,:]
                
            k[i,:] = F(Up, t + c[i]*dt)
            
        U2 = U1 + dt*matmul(bs,k)

    return U2


def StepSize(dU, tol, dt, orders): 
    error = norm(dU)

    if error > tol:
        step_size = dt*(tol/error)**(1/(orders+1))
    else:
        step_size = dt

    return step_size


# MATRIZ DE BUTCHER
def Matriz_Butcher():

    orders = [2, 1]
    Ns = 2

    a = zeros([Ns, Ns-1])
    b = zeros([Ns])
    bs = zeros([Ns])
    c = zeros([Ns])

    c = [0., 1.]
    a[0, :] = [0.]
    a[1, :] = [1.]
    b[:] = [1./2, 1./2]
    bs[:] = [1., 0.]

    return orders, Ns, a, b, bs, c



#===============================================================================
# METODO ADAMS-BASHFORTH DE ORDEN 4

def G(U, t, dt, F):

    N = len(U)
    t_old = 0

    if t < t_old or t_old == 0:
        U2 = U + dt * F(U,t)
    else:
        U2 = U0 + 2 * dt * F(U,t)
        U0 = U

    t_old = t + dt
    return U2

def AB4_(U, t, dt, F):
    N = len(U)
    history = []

    if len(history) < 4:
        # Utiliza Euler para los primeros pasos si no hay suficientes datos
        if len(history) == 0:
            history.append(U)
        for _ in range(3 - len(history)):
            U_next = U + dt * F(U, t)
            history.append(U_next)
            U = U_next
            t += dt

        return U_next

    # Adams-Bashforth de cuarto orden
    f0 = F(history[-1], t - 3 * dt)
    f1 = F(history[-2], t - 2 * dt)
    f2 = F(history[-3], t - dt)
    f3 = F(history[-4], t)

    U_next = U + (dt / 24) * (55 * f3 - 59 * f2 + 37 * f1 - 9 * f0)
    history.append(U_next)
    return U_next


def AB4(U, t, dt, F):
 
    N = len(U)
    if t == 0:
    
        global previous_U 
        global previous_U_state
        previous_U = zeros([3,N])
        previous_U_state = zeros([3,1])
 
    # Si hay resultados anteriores faltantes, utiliza Euler para calcularlos
    if any(previous_U_state == 0):
        for ii in range(3):
            if previous_U_state[ii] == 0:
                # Calcula los primeros pasos con el método de Euler
                previous_U[ii,:] = F(U, t)
                previous_U_state[ii] = 1
                return (U + dt * previous_U[ii,:]) # Devuelve el resultado final después de los primeros pasos
 
    # Almacena los valores de previous_U en unas variables auxiliares
    Un3 = previous_U[0,:]
    Un2 = previous_U[1,:]
    Un1 = previous_U[2,:]
 
    # Adams-Bashforth de cuarto orden
    f0 = Un3
    f1 = Un2
    f2 = Un1
    f3 = F(U, t)
 
    # Actualiza el historial eliminando el resultado más antiguo y agregando el nuevo
    previous_U[0,:] = Un2
    previous_U[1,:] = Un1
    previous_U[2,:] = f3
 
    return U + (dt / 24) * (55 * f3 - 59 * f2 + 37 * f1 - 9 * f0)