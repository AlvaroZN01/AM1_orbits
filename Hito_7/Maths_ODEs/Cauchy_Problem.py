
#=================================================================================
#								  CAUCHY PROBLEM
#=================================================================================

# Dado un Problema de Valor Inicial (problema de Cauchy)
#
#		dU(t)/dt = F(t,U(t))
#		U(t_0) = U_0

from numpy import array, zeros, linspace

def Cauchy_Problem( F, t, U_0, Temporal_Scheme ): 

    N =  len(t)-1
    N_var = len(U_0)
    U = zeros( ( N+1, N_var ) )

    U[ 0, : ] = U_0

    for n in range(N):

        U[ n+1, : ] = Temporal_Scheme( U[ n, : ], t[ n ], t[ n+1 ] - t[ n ],  F )

    return U