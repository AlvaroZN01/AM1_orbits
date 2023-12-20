from numpy import array

# Ecuacion de un oscilador
def Oscillator(t, U):
    return array([U[1], -U[0]])