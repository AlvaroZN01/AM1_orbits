
#=================================================================================
#								     KEPLER
#=================================================================================

#=================================================================================
# PROBLEMA DE KEPLER
#
# En Mecanica Clasica, el Problema de Kepler es un caso especial del Problema de los Dos Cuerpos, en el que los dos cuerpos 
# interactuan por medio de una fuerza central que varia en intensidad según una ley cuadratica inversamente proporcional en 
# funcion de la distancia entre ambos.
#
#		d2r / dt2 = - r / |r|^3 
#		r (t = 0)
#		dr / dt (t = 0)
#
# Notese que r es un vector de dos componentes, x e y. Al ser la ecuacion de segundo orden (por tener una derivada segunda),
# en primer lugar es necesario reducir el problema a un sistema de primer orden.
#		
#		r = ( x , y )
#		|r| = ( x^2 + y^2 )^(1/2)		
#		dr/dt = ( vx , vy )
#		U = ( r , dr/dt )
#		dU/dt = ( dr/dt , d2r/dt2 ) = ( dr/dt , -r/|r|^3 ) = F ( t , U(t) )
# 
# Se puede ver que en este caso no hay una dependencia explicita con el tiempo. Al vector U se le denomina vector de estado.

from numpy import array

def Kepler_Equation( U, t ):

	x = U[0]			# La coordenada x es la primera componente del vector de estado.
	y = U[1]			# La coordenada y es la segunda componente del vector de estado.
	dx_dt = U[2]		# La velocidad según x es la tercera componente del vector de estado.
	dy_dt = U[3]		# La velocidad según y es la cuarta componente del vector de estado.

	mod_r = ( x**2 + y**2 )**0.5

	return array( [ dx_dt , dy_dt , -x/mod_r**3 , -y/mod_r**3 ] )
