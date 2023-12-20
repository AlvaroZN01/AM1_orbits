
#=================================================================================
#								     CR3BP
#=================================================================================

from numpy import array, zeros, size

# CALCULO DEL JACOBIANO
def Jacobiano(F, U):

	N = size(U)
	Jacobian = zeros([N,N])
	t = 1e-10

	for i in range(N):

		xj = zeros(N)
		xj[i] = t
		Jacobian[:,i] = (F(U + xj) - F(U - xj))/(2*t)

	return Jacobian
