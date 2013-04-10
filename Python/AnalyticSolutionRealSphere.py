# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:24:40 2013

@author: Jan
"""


'''PROBLEM: Sphere is 3D, current model is just 2D'''



'''IMPORT'''
from pylab import *
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.ticker import FormatStrFormatter

'''CONSTANTS'''
PHI_0 = 2067.83  # magnetic flux in T*nm²

'''INPUT VARIABLES'''
b0 = 1.0 #in T
v0 = 0
Vacc = 30000
xDim, yDim = 160, 160
res = 10 #in nm
beta = 300.0/360.0 * 2*pi
x0 = res*xDim*0.5
y0 = res*yDim*0.5
R = res*xDim*0.2

'''CREATE COORDINATE GRIDS'''
x = np.linspace(res/2,xDim*res-res/2,num=xDim)
y = np.linspace(res/2,yDim*res-res/2,num=yDim)
xx, yy = np.meshgrid(x,y)

'''COMPUTATION MAGNETIC PHASE SHIFT (REAL SPACE) DISC'''
def F0(x,y):
    a = np.log( x**2 + y**2 )
    b = np.arctan2( x, y )  # atan or atan2?
    return x*a - 2*x + 2*y*b   
  
coeff = - 2/3 * pi * R**3 * b0 / PHI_0 * res/R
  
def phiMag(x,y):
    r = np.sqrt((x-x0) ** 2 + (y-y0) ** 2)      
    result = coeff  / r**2 * ((y-y0) * np.cos(beta) - (x-x0) * np.sin(beta))
    in_or_out = 1 * (r > R) + (1 - (1-(r/R)**2)**(3/2)) * (r < R)    
    result *= in_or_out
    return result

phaseMag = phiMag(xx, yy)
    
def plot_phase(func, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    plt.pcolormesh(func, cmap='Greys')
    ticks = ax.get_xticks() * res
    ax.set_xticklabels(ticks.astype(int))
    ticks = ax.get_yticks() * res
    ax.set_yticklabels(ticks.astype(int))
    ax.set_title('Analytical Solution' + name)
    ax.set_xlabel('x-axis [nm]')
    ax.set_ylabel('y-axis [nm]')
    plt.colorbar()
    plt.show()
    
plot_phase(phaseMag, ' Sphere - Phase')
plot_phase(cos(16*phaseMag), ' Sphere - Cos(Phase)')