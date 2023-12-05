from numpy import zeros, linspace

def Stability_region(Scheme, N, x0, xf, y0, yf): 
    x = linspace(x0, xf, N)
    y = linspace(y0, yf, N)
    rho =  zeros((N, N))

    for i in range(N): 
      for j in range(N):
          w = complex(x[i], y[j])
          r = Scheme( 1., 1., 0., lambda u, t: w*u )
          rho[i, j] = abs(r) 

    return rho, x, y  