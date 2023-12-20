
#=================================================================================
#								     CR3BP
#=================================================================================

from Maths_ODEs.Cauchy_Problem import Cauchy_Problem
from Maths_ODEs.Temporal_Schemes import Explicit_Euler, Inverse_Euler, Crank_Nicolson, RungeKutta_4, RungeKutta_Embedded
from Maths_Equations.Eq_Systems import Jacobiano

from numpy import zeros_like, sqrt, sum, square, array, zeros
from scipy.optimize import fsolve
from numpy.linalg import eig


# PROBLEMA RESTRINGIDO DE LOS TRES CUERPOS
def CR3BP(U, REL):

    x = U[0]
    y = U[1]
    r = array([x,y])

    dxdt = U[2]
    dydt = U[3] 
    drdt = array([dxdt,dydt])

    r1 = sqrt((x + REL)**2 + y**2)
    r2 = sqrt((x - 1 + REL)**2 + y**2)

    dvdt1 = - (1 - REL)*(x + REL)/(r1**3) - REL*(x - 1 + REL)/(r2**3)
    dvdt2 = - (1 - REL)*y/(r1**3) - REL*y/(r2**3)

    return array([ dxdt, dydt, 2*dydt + x + dvdt1, -2*dxdt + y + dvdt2])

   

# PUNTOS DE LAGRANGE
def Puntos_Lagrange(U_0, Np, REL):
    
    Lagrange_Points = zeros([5,2])

    def F(U):
        X = zeros(4)
        X[0:2] = U
        X[2:4] = 0
        FX = CR3BP(X, REL)
        return FX[2:4]
   
    for i in range (Np):

        Lagrange_Points[i,:] = fsolve(F, U_0[i,0:2])

    return Lagrange_Points



# ESTABILIDAD DE LOS PUNTOS DE LAGRANGE
def Estabilidad_Lagrange(U_0, REL):

    def F(U):
        return CR3BP(U, REL)

    J = Jacobiano(F, U_0)
    Autovalores, Autovectores = eig(J)

    return Autovalores





