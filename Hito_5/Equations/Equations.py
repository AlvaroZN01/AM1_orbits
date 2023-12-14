from numpy import array, reshape, zeros, concatenate, split
from numpy.linalg import norm

def F_Kepler(t, U):
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    mr = (x**2 + y**2)**1.5
    return array([vx, vy, -x/mr, -y/mr])

def Oscillator(t, U):
    return array([U[1], -U[0]])

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