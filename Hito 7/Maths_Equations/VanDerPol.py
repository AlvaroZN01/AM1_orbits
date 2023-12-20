
#=================================================================================
#								 VAN DER POL
#=================================================================================

from Maths_Equations.Box_Muller import Box_Muller_random_generator
from numpy import array, cos, sin, arange, sqrt, pi
from numpy.random import randn


def VanDerPol_Libre( U, t ):

	x = U[0]			# La coordenada x es la primera componente del vector de estado.
	y = U[1]			# La coordenada y es la segunda componente del vector de estado.

	return array( [y , 1*(1 - x**2)*y - x  ] )

def VanDerPol_Libre2( U, t ):

	x = U[0]			# La coordenada x es la primera componente del vector de estado.
	y = U[1]			# La coordenada y es la segunda componente del vector de estado.

	return array( [y , 2*(1 - x**2)*y - x  ] )

def VanDerPol_Libre3( U, t ):

	x = U[0]			# La coordenada x es la primera componente del vector de estado.
	y = U[1]			# La coordenada y es la segunda componente del vector de estado.

	return array( [y , 4*(1 - x**2)*y - x  ] )

def VanDerPol_Libre4( U, t ):

	x = U[0]			# La coordenada x es la primera componente del vector de estado.
	y = U[1]			# La coordenada y es la segunda componente del vector de estado.

	return array( [y , 0.5*(1 - x**2)*y - x  ] )


def VanDerPol_ForzadoArmonico( U, t ):

	x = U[0]			# La coordenada x es la primera componente del vector de estado.
	y = U[1]			# La coordenada y es la segunda componente del vector de estado.

	return array( [y , 4 * (1 - x**2) * y - 1 * x + 2 *cos(pi*t)  ] )

def VanDerPol_ForzadoArmonico2( U, t ):

	x = U[0]			# La coordenada x es la primera componente del vector de estado.
	y = U[1]			# La coordenada y es la segunda componente del vector de estado.

	return array( [y , -8.53 * (1 - x**2) * y - 1 * x + 1.2*cos(pi/5*t)  ] )



def VanDerPol_ForzadoEstocastico(U, t ):

    mu = 0
    tau = 0.1
    D = 100
    h = 0.01;	x = U[0]; y = U[1]; xi = U[2]
    Z = Box_Muller_random_generator(2,1)
 
    return array([ y, mu*(1-x**2)*y - x + xi, -xi/tau + sqrt(D)*sqrt(h)*Z[0,0]])

def VanDerPol_ForzadoEstocastico2(U, t ):

    mu = 0
    tau = 0.1
    D = 200000
    h = 0.01;	x = U[0]; y = U[1]; xi = U[2]
    Z = Box_Muller_random_generator(2,1)
 
    return array([ y, mu*(1-x**2)*y - x + xi, -xi/tau + sqrt(D)*sqrt(h)*Z[0,0]])

def VanDerPol_ForzadoEstocastico3(U, t ):

    mu = 0
    tau = 0.1
    D = 20000
    h = 0.01;	x = U[0]; y = U[1]; xi = U[2]
    Z = Box_Muller_random_generator(2,1)
 
    return array([ y, mu*(1-x**2)*y - x + xi, -xi/tau + sqrt(D)*sqrt(h)*Z[0,0]])