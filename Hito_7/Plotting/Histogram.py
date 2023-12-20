
#=================================================================================
#								    HISTOGRAM
#=================================================================================

import matplotlib.pyplot as plt
from numpy import array, zeros, linspace, abs, transpose, float64, histogram2d, meshgrid, ones_like, max, min
from mpl_toolkits.mplot3d import Axes3D


def Histogram_2D(Data_X, Data_Y, Label_X, Label_Y, save_name):

    plt.figure(figsize=(7, 7))
    plt.hist2d(Data_X, Data_Y, bins=50, cmap='viridis')
    plt.colorbar(label='Frecuencia')
    plt.xlabel(Label_X)
    plt.ylabel(Label_Y)
    plt.grid(True)
    plt.savefig(save_name)
    plt.show()



def Histogram_3D(Data_X, Data_Y, Label_X, Label_Y):
    
    plt.figure(figsize=(7, 7))
    plt.subplot(111, projection='3d')

    hist, xedges, yedges = histogram2d(Data_X, Data_Y, bins=30)

    xpos, ypos = meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    dx = dy = 0.5 * ones_like(zpos)
    dz = hist.ravel()

    bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', cmap='viridis')

    plt.xlabel(Label_X)
    plt.ylabel(Label_Y)
    plt.zlabel('Frecuencia')

    plt.show()
