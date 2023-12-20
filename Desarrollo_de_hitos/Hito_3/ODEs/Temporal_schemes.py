from scipy import optimize

# Esquema temporal Euler
def Euler(U0, t0, tf, f):
    return U0 + (tf - t0) * f(t0, U0)

# Esquema temporal Crank Nicolson
def Crank_Nicolson(U0, t0, tf, f):
    def Residual(x):
        return x - U0 - (tf - t0)/2 * (f(t0, U0) + f(tf, x))
    return optimize.newton(func = Residual, x0 = U0, maxiter = 250)

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
    return optimize.newton(func = Residual, x0 = U0, maxiter = 250)