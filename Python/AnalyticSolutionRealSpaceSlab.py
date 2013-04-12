# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:24:40 2013

@author: Jan
"""

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
Lx = res*xDim*0.2
Ly = res*yDim*0.4

'''CREATE COORDINATE GRIDS'''
x = np.linspace(res/2,xDim*res-res/2,num=xDim)
y = np.linspace(res/2,yDim*res-res/2,num=yDim)
#xx, yy = np.meshgrid(x,y)

'''COMPUTATION MAGNETIC PHASE SHIFT (REAL SPACE) SLAB'''
def F0(x,y):
    a = np.log( x**2 + y**2 )
    b = np.arctan( x / y )  # atan or atan2?
    return x*a - 2*x + 2*y*b   
  
coeff = b0 * res / ( 4 * PHI_0 )
  
def phiMag(x,y):
    return coeff * ( - np.cos(beta) * ( F0(x-x0-Lx/2,y-y0-Ly/2)-F0(x-x0+Lx/2,y-y0-Ly/2)
                                       -F0(x-x0-Lx/2,y-y0+Ly/2)+F0(x-x0+Lx/2,y-y0+Ly/2) )
                     + np.sin(beta) * ( F0(y-y0-Ly/2,x-x0-Lx/2)-F0(y-y0+Ly/2,x-x0-Lx/2)
                                       -F0(y-y0-Ly/2,x-x0+Lx/2)+F0(y-y0+Ly/2,x-x0+Lx/2) ) )

def F0_Cos_1(x,y):
    return coeff * np.cos(beta) * F0(x-x0-Lx/2,y-y0-Ly/2)
    
def F0_Cos_2(x,y):
    return coeff * np.cos(beta) * F0(x-x0+Lx/2,y-y0-Ly/2)
    
def F0_Cos_3(x,y):
    return coeff * np.cos(beta) * F0(x-x0-Lx/2,y-y0+Ly/2)
    
def F0_Cos_4(x,y):
    return coeff * np.cos(beta) * F0(x-x0+Lx/2,y-y0+Ly/2)
    
def F0_Sin_1(x,y):
    return coeff * np.sin(beta) * F0(y-y0-Ly/2,x-x0-Lx/2)
    
def F0_Sin_2(x,y):
    return coeff * np.sin(beta) * F0(y-y0+Ly/2,x-x0-Lx/2)
    
def F0_Sin_3(x,y):
    return coeff * np.sin(beta) * F0(y-y0-Ly/2,x-x0+Lx/2)
    
def F0_Sin_4(x,y):
    return coeff * np.sin(beta) * F0(y-y0+Ly/2,x-x0+Lx/2)


def phiMag1(x,y):
    return coeff * ( - 0*np.cos(beta) + 1*( F0(x-x0-Lx/2,y-y0-Ly/2)-F0(x-x0+Lx/2,y-y0-Ly/2)) )
                                       
def phiMag2(x,y):
    return coeff * ( - 0*np.cos(beta) + 1*(-F0(x-x0-Lx/2,y-y0+Ly/2)+F0(x-x0+Lx/2,y-y0+Ly/2) ) )

def phiMag3(x,y):
    return coeff * ( + 0*np.sin(beta) + 1*( F0(y-y0-Ly/2,x-x0-Lx/2)-F0(y-y0+Ly/2,x-x0-Lx/2) ) )

def phiMag4(x,y):
    return coeff * ( + 0*np.sin(beta) + 1*(-F0(y-y0-Ly/2,x-x0+Lx/2)+F0(y-y0+Ly/2,x-x0+Lx/2) ) )                                     
                                     
def Test(x,y):
    return ( abs(x-x0)<Lx/2 and abs(y-y0)<Ly/2 )

xx, yy = np.meshgrid(x,y)

phaseMag = phiMag(xx, yy)

phase_F0_Cos_1 = F0_Cos_1(xx, yy)
phase_F0_Cos_2 = F0_Cos_2(xx, yy)
phase_F0_Cos_3 = F0_Cos_3(xx, yy)
phase_F0_Cos_4 = F0_Cos_4(xx, yy)
phase_F0_Sin_1 = F0_Sin_1(xx, yy)
phase_F0_Sin_2 = F0_Sin_2(xx, yy)
phase_F0_Sin_3 = F0_Sin_3(xx, yy)
phase_F0_Sin_4 = F0_Sin_4(xx, yy)

sinphase_F0_Cos_1 = np.sin(phase_F0_Cos_1)
sinphase_F0_Cos_2 = np.sin(phase_F0_Cos_2)
sinphase_F0_Cos_3 = np.sin(phase_F0_Cos_3)
sinphase_F0_Cos_4 = np.sin(phase_F0_Cos_4)
sinphase_F0_Sin_1 = np.sin(phase_F0_Sin_1)
sinphase_F0_Sin_2 = np.sin(phase_F0_Sin_2)
sinphase_F0_Sin_3 = np.sin(phase_F0_Sin_3)
sinphase_F0_Sin_4 = np.sin(phase_F0_Sin_4)

phase_F0_Cos_Complete = (phase_F0_Cos_1 + phase_F0_Cos_2
                       + phase_F0_Cos_3 + phase_F0_Cos_4)
                         
phase_F0_Sin_Complete = (phase_F0_Sin_1 + phase_F0_Sin_2
                       + phase_F0_Sin_3 + phase_F0_Sin_4)

sinphase_F0_Cos_Complete = (sinphase_F0_Cos_1 + sinphase_F0_Cos_2
                          + sinphase_F0_Cos_3 + sinphase_F0_Cos_4)
                         
sinphase_F0_Sin_Complete = (sinphase_F0_Sin_1 + sinphase_F0_Sin_2
                          + sinphase_F0_Sin_3 + sinphase_F0_Sin_4)

phaseMag1 = phiMag1(xx, yy)
phaseMag2 = phiMag2(xx, yy)
phaseMag3 = phiMag3(xx, yy)
phaseMag4 = phiMag4(xx, yy)
    
    
def plot_phase(func, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    plt.pcolormesh(func, cmap='jet')
    ticks = ax.get_xticks() * res
    ax.set_xticklabels(ticks.astype(int))
    ticks = ax.get_yticks() * res
    ax.set_yticklabels(ticks.astype(int))
    ax.set_title('Analytical Solution' + name)
    ax.set_xlabel('x-axis [nm]')
    ax.set_ylabel('y-axis [nm]')
    plt.colorbar()
    plt.show()
    

'''PLOT'''

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
plt.pcolormesh(np.cos(16*phaseMag), cmap='Greys')
ticks = ax.get_xticks()*res
ax.set_xticklabels(ticks.astype(int))
ticks = ax.get_yticks()*res
ax.set_yticklabels(ticks.astype(int))
ax.set_title('Analytical Solution')
ax.set_xlabel('x-axis [nm]')
ax.set_ylabel('y-axis [nm]')
plt.colorbar()
plt.show()

'''PLOT F0s'''
plot_phase(phase_F0_Cos_Complete, 'F0_Cos_Complete')
plot_phase(phase_F0_Sin_Complete, 'F0_Sin_Complete')
plot_phase(sinphase_F0_Cos_Complete, 'Sin_F0_Cos_Complete')
plot_phase(sinphase_F0_Sin_Complete, 'Sin_F0_Sin_Complete')
#plot_phase(phase_F0_Cos_1, 'F0_Cos_1')
#plot_phase(phase_F0_Cos_2, 'F0_Cos_2')
#plot_phase(phase_F0_Cos_3, 'F0_Cos_3')
#plot_phase(phase_F0_Cos_4, 'F0_Cos_4')
#plot_phase(phase_F0_Sin_1, 'F0_Sin_1')
#plot_phase(phase_F0_Sin_2, 'F0_Sin_2')
#plot_phase(phase_F0_Sin_3, 'F0_Sin_3')
#plot_phase(phase_F0_Sin_4, 'F0_Sin_4')
#plot_phase(phase_F0_Cos_1-phase_F0_Cos_3, 'F0_Cos_1-3')
#plot_phase(phase_F0_Cos_2-phase_F0_Cos_4, 'F0_Cos_2-4')
#plot_phase(phase_F0_Sin_1-phase_F0_Sin_3, 'F0_Sin_1-3')
#plot_phase(phase_F0_Sin_2-phase_F0_Sin_4, 'F0_Sin_2-4')
#plot_phase(phase_F0_Cos_1-phase_F0_Cos_3 + phase_F0_Cos_2-phase_F0_Cos_4,
#           'F0_Cos_Complete')
#plot_phase(phase_F0_Sin_1-phase_F0_Sin_3 + phase_F0_Sin_2-phase_F0_Sin_4,
#           'F0_Sin_Complete')
#           plot_phase(phase_F0_Sin_1-phase_F0_Sin_3 + phase_F0_Sin_2-phase_F0_Sin_4,
#           'F0_Sin_Complete')


x = np.linspace(-50,50,100)
y = np.linspace(-50,50,100)

xx, yy = np.meshgrid(x,y)

Arc2 = np.arctan2(yy,xx)
Arc = np.arctan(yy/xx)