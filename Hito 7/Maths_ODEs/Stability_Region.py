
#=================================================================================
#								 STABILITY REGION
#=================================================================================

from numpy import zeros, linspace, abs, float64

def Region_Estabilidad(Scheme, N, x0, xf, y0, yf): 

    x = linspace( x0, xf, N )
    y = linspace( y0, yf, N )
    rho =  zeros( (N, N),  dtype=float64)

    for i in range(N): 
        for j in range(N):

            w = complex(x[i], y[j])
            r = Scheme( 1., 0., 1., lambda u, t: w*u )
            rho[i, j] = abs(r) 

    return rho, x, y  
