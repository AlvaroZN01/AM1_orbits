from numpy import array, reshape, zeros, concatenate, split, sqrt, linspace
from numpy.linalg import norm

# Ecuacion de las orbitas de Kepler
def F_Kepler(t, U):
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    mr = (x**2 + y**2)**1.5
    return array([vx, vy, -x/mr, -y/mr])

# Problema de los N cuerpos
def N_body(t, U, Nb, Nc):      
    Us = reshape(U, (Nb, Nc, 2))   
    r = reshape(Us[:, :, 0], (Nb, Nc))
    v = reshape(Us[:, :, 1], (Nb, Nc))

    F = zeros(len(U))   
    dUs = reshape(F, (Nb, Nc, 2))  
    drdt = reshape(dUs[:, :, 0], (Nb, Nc))
    dvdt = reshape(dUs[:, :, 1], (Nb, Nc))
    
    dvdt[:,:] = 0

    for i in range(Nb):   
        drdt[i,:] = v[i,:]
        for j in range(Nb): 
            if j != i:  
                d = r[j,:] - r[i,:]
                dvdt[i,:] = dvdt[i,:] +  d[:] / norm(d)**3 
    
    return F

# Problema de los 3 cuerpos circular restringido
def N3_body_restricted(t, U, m1, m2, r12):
    x, y, vx, vy = U[0], U[1], U[2], U[3]

    G = 6.6743e-20 # [N km^2 kg^-2]

    omega = sqrt(G*(m1+m2)/(r12**3))
    pi1 = m1 / (m1 + m2)
    pi2 = m2 / (m1 + m2)
    mu1 = G * m1
    mu2 = G * m2

    x1 = -r12*pi2
    x2 = r12*pi1

    r1 = sqrt((x - x1)**2 + y**2)
    r2 = sqrt((x - x2)**2 + y**2)
    
    dotdotx = omega**2 * x + 2*omega*vy - (mu1/r1**3) * (x + pi2*r12) - (mu2/r2**3) * (x - pi1*r12)
    dotdoty = omega**2 * y - 2*omega*vx - (mu1/r1**3) * y - (mu2/r2**3) * y

    return array([vx, vy, dotdotx, dotdoty])

def Stability_Lagrange(x, y, m1, m2, r12):
    # x = linspace(lims_malla[0,0], lims_malla[0,1], 100)
    # y = linspace(lims_malla[1,0], lims_malla[1,1], 100)

    pi1 = m1 / (m1 + m2)
    pi2 = m2 / (m1 + m2)

    U = zeros([len(x), len(y)])
    for ii in range(0, len(x)):
        for jj in range(0, len(y)):
            sigma = sqrt((x[ii]/r12 + pi2)**2 + (y[jj]/r12)**2)
            psi = sqrt((x[ii]/r12 - pi1)**2 + (y[jj]/r12)**2)
            U[ii,jj] = -pi1/sigma - pi2/psi - 1/2 * (pi1*sigma**2 + pi2*psi**2)

    return(U)
