from numpy import array, zeros, linalg
import matplotlib.pyplot as plt
from scipy import optmize


def F_Kepler(U):
    
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    mr = (x**2 + y**2)**1.5
    return array([vx, vy, -x/mr, -y/mr])

def CN(U):
    return U + dt/2 * (F_Kepler(U) + F_Kepler(CN(U)))    


N = 10000
#U = [1, 0, 0, 1]
#print(type(U))

U = array([1, 0, 0, 1])
print(type(U))

dt = 0.001

x = array(zeros(N))
y = array(zeros(N))
x[0] = U[0]
y[0] = U[1]

for ii in range(1,N):
#    F = array([U[2], U[3], -U[0]/(U[0]**2+U[1]**2)**1.5, -U[1]/(U[0]**2+U[1]**2)**1.5])
    F = F_Kepler(U)
    U = U + dt * F
    x[ii] = U[0]
    y[ii] = U[1]

plt.axis('equal')
plt.plot(x,y)
plt.show()

##############################

U = array([1, 0, 0, 1])
x = array(zeros(N))
y = array(zeros(N))
j = array(zeros(N))
U_aux1 = array(zeros(4))
U_aux2 = array(zeros(4))
x[0] = U[0]
y[0] = U[1]

for ii in range(1,N):
    U_aux1 = U
    for jj in range (1,50):
        U_aux2 = U + dt/2 * (F_Kepler(U) + F_Kepler(U_aux1))
        if abs(linalg.norm(U_aux2) - linalg.norm(U_aux1)) < 1e-12:
            U = U_aux2
            j[ii] = jj
            break
        else:
            U_aux1 = U_aux2
            
    x[ii] = U[0]
    y[ii] = U[1]

# optimize.newton
        
plt.axis('equal')
plt.plot(x,y)
plt.show()  

plt.plot(range(0,N), j)
plt.show()

#############################

U = array([1, 0, 0, 1])
x = array(zeros(N))
y = array(zeros(N))
x[0] = U[0]
y[0] = U[1]

for ii in range(1,N):
    k1 = F_Kepler(U)
    k2 = F_Kepler(U + k1*dt/2)
    k3 = F_Kepler(U + k2*dt/2)
    k4 = F_Kepler(U + k3*dt)
    U = U + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    x[ii] = U[0]
    y[ii] = U[1]

plt.axis('equal')
plt.plot(x,y)
plt.show()

