from numpy import array, zeros

# Función para integrar ecuaciones (problema de Cauchy)
def Cauchy(t, temporal_scheme, f, U0):
    U = array (zeros((len(U0),len(t)))) # Definicion del tamaño de la solucion
    U[:,0] = U0
    for ii in range(0, len(t) - 1):
        U[:,ii+1] = temporal_scheme(U[:,ii], t[ii], t[ii+1], f) # Integracion empleando el esquema numerico indicado
    return U