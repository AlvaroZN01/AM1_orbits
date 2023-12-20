
print( "" )
print( "   ==========================================================================" )
print( "   ==========================================================================" )
print( "   ==           MASTER UNIVERSITARIO EN SISTEMAS ESPACIALES                ==" )
print( "   ==                         IDR - ETSIAE - UPM                           ==" )
print( "   ==                    AMPLIACION DE MATEMATICAS 1                       ==" )
print( "   ==                                                                      ==" )
print( "   ==                      Alejandro Fernandez Ruiz                        ==" )
print( "   ==                      Ruben Garrido Echevarria                        ==" )
print( "   ==                        Javier Pueyo Serrano                          ==" )
print( "   ==                        Alvaro Zurdo Navajo                           ==" )
print( "   ==========================================================================" )
print( "   ==========================================================================" )
print( "" )
print( "" )
print( "   ==========================================================================" )
print( "   =========================     TRABAJO FINAL     ==========================" )
print( "   ==========================================================================" )
print( "" )

from Maths_ODEs.Temporal_Schemes import Explicit_Euler, Inverse_Euler, Crank_Nicolson, RungeKutta_4, RungeKutta_EMB, AB4
from Maths_ODEs.Cauchy_Problem import Cauchy_Problem
from Maths_ODEs.Stability_Region import Region_Estabilidad
from Maths_ODEs.Temporal_Error import Error_Cauchy_Problem, Temporal_Convergence_Rate

from Maths_Equations.VanDerPol import VanDerPol_Libre, VanDerPol_ForzadoArmonico, VanDerPol_ForzadoEstocastico, VanDerPol_Libre2, VanDerPol_Libre3,VanDerPol_ForzadoEstocastico2
from Maths_Equations.Stochastic import Proceso_Wiener, Proceso_OrnsteinUhlenbeck
from Physics.Kepler_Equation import Kepler_Equation

from Plotting.Graphics import Plot_2D, Plot_3D
from Plotting.Histogram import Histogram_2D, Histogram_3D
from Plotting.Animation import create_animation


import matplotlib.pyplot as plt
from numpy import array, zeros, linspace, abs, transpose, float64, histogram2d, meshgrid, ones_like, max, min
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Graficos con tipografia LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})



tf = 100
N = 10000
print( "" )
t = linspace( start = 0, stop = tf, num = N)
U_0 = array([1,1,0.1])
U1 =  Cauchy_Problem( VanDerPol_ForzadoEstocastico, t, U_0, AB4 )
U_01 = array([0.1,0.1])
U2 =  Cauchy_Problem( VanDerPol_ForzadoEstocastico2, t, U_0, AB4 )
#U3 =  Cauchy_Problem( VanDerPol_Libre3, t, U_0, AB4 )
#print(U1)
#Plot_2D(U1[:10000,0],U1[:10000,1],'x','y')

#Plot_2D(t[:10000],U1[:10000,0],'x','y')
#plt.show()
##plt.show()
#plt.figure(figsize=(7,7))
#plt.plot(U1[:,0],U1[:,1], color='b',label='$\mu=1$')
##plt.plot(U2[:,0],U2[:,1], color='r',label='$\mu=2$')
##plt.plot(U3[:,0],U3[:,1], color='g',label='$\mu=4$')
##plt.axis('equal')
#plt.xlabel('$x$',fontsize=14)
#plt.ylabel('$dx/dt$',fontsize=14)
#plt.grid(True)
#plt.legend(fontsize=14)
#plt.savefig('VDP_mu_positivos.png')
#plt.show()
create_animation(U1,15,15,'VDPL_est_0.gif',N)
create_animation(U2,15,15,'VDPL_est_1.gif',N)

#Proceso_OrnsteinUhlenbeck(1, 3, 1, 1000)
##plt.xlabel('$x$',fontsize=14)
##plt.ylabel('$dx/dt$',fontsize=14)
#plt.grid(False)
#plt.savefig('Ornstein_Uhlenbeck.png')
#plt.show()