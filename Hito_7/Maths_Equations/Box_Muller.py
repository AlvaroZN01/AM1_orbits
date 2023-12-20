
#=================================================================================
#								 BOX-MULLER ALGORITHM
#=================================================================================

from numpy import zeros, sqrt, log, cos, pi
from random import random

def Box_Muller_random_generator(a, b):

        # Genera un número aleatorio con distribución normal estándar utilizando el método de Box-Muller
        # Input: Dimensiones del output (hasta 2 dimensiones)
        # Output: Numero aleatorio con distribucion normal estandar
        Z1 = zeros((a, b))
        for i in range(0, a):
            for j in range(0, b):
                U1 = 1.0 - random()
                U2 = 1.0 - random()
                Z1[i, j] = sqrt(-2*log(U1)) * cos(2*pi*U2)
                
        return Z1
