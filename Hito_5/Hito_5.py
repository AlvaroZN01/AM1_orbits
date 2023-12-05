from numpy import arange, array, log10, linalg, zeros, vstack, ones, linspace, transpose
from numpy.linalg import norm, lstsq
import matplotlib.pyplot as plt
from ODEs.Cauchy_problem import Cauchy
from ODEs.Temporal_schemes import Euler, Inverse_Euler, RK4, Crank_Nicolson
from ODEs.Temporal_error import Error_Cauchy_Problem, Convergence_rate
from ODEs.Stability import Stability_region

# Function to be integrated
# def N_body()