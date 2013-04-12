# -*- coding: utf-8 -*-
"""Create and display a phase map from magnetization data."""


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi


PHI_0 = 2067.83  # magnetic flux in T*nm²


def fourier_space(mag_data, b_0=1, padding=0, v_0=0, v_acc=30000):
    '''Calculate phasemap from magnetization data (fourier approach).
    Arguments:
        mag_data - MagDataLLG object (from magneticimaging.dataloader) storing
                   the filename, coordinates and magnetization in x, y and z
        b_0      - magnetic induction corresponding to a magnetization Mo in T 
                   (default: 1)
        padding  - factor for zero padding, the default is 0 (no padding), for
                   a factor of n the number of pixels is increase by (1+n)**2,
                   padded zeros are cropped at the end
        v_0      - average potential of the sample in V (default: 0)
        v_acc    - acceleration voltage of the microscop in V (default: 30000)
    Returns:
        the phasemap as a 2 dimensional array
    
    '''    
    res  = mag_data.res
    x_dim, y_dim, z_dim = mag_data.dim
    x_mag, y_mag, z_mag = mag_data.magnitude
    
    # TODO: interpolation (redefine xDim,yDim,zDim) thickness ramp
    
    x_pad = x_dim/2 * padding
    y_pad = y_dim/2 * padding
    x_mag_big = np.zeros(((1 + padding) * y_dim, (1 + padding) * x_dim))
    y_mag_big = np.zeros(((1 + padding) * y_dim, (1 + padding) * x_dim))
    x_mag_big[y_pad : y_pad + y_dim, x_pad : x_pad + x_dim] = x_mag
    y_mag_big[y_pad : y_pad + y_dim, x_pad : x_pad + x_dim] = y_mag
    
    x_mag_fft = np.fft.fftshift(np.fft.rfft2(x_mag_big), axes=0)
    y_mag_fft = np.fft.fftshift(np.fft.rfft2(y_mag_big), axes=0)
    nyq = 1 / res  # nyquist frequency
    x = np.linspace(      0, nyq/2, x_mag_fft.shape[1])
    y = np.linspace( -nyq/2, nyq/2, x_mag_fft.shape[0]+1)[:-1]
    xx, yy = np.meshgrid(x, y)
                         
    phase_fft = (1j * b_0) / (2 * PHI_0) * ((x_mag_fft * yy - y_mag_fft * xx) 
                                           / (xx ** 2 + yy ** 2 + 1e-18))
    phase_big = np.fft.irfft2(np.fft.ifftshift(phase_fft, axes=0))
    
    phase = phase_big[y_pad : y_pad + y_dim, x_pad : x_pad + x_dim]
    
    # TODO: Electrostatic Component

    return phase
    
    
def real_space(mag_data, b_0=1, v_0=0, v_acc=30000):
    '''Calculate phasemap from magnetization data (real space approach).
    Arguments:
        mag_data - MagDataLLG object (from magneticimaging.dataloader) storing
                   the filename, coordinates and magnetization in x, y and z
    Returns:
        the phasemap as a 2 dimensional array
        
    '''
    # TODO: Expand docstring!
    
    res  = mag_data.res
    x_dim, y_dim, z_dim = mag_data.dim
    x_mag, y_mag, z_mag = mag_data.magnitude    
    
    beta = np.arctan2(y_mag, x_mag)

    abs_mag = np.sqrt(x_mag**2 + y_mag**2)    
     
    coeff = abs_mag * res / ( 4 * PHI_0 ) / res # TODO: Lz = res 
     
    def F0(x,y):
        a = np.log(x**2 + y**2)
        b = np.arctan(x / (y + 1E-18) )
        return x*a - 2*x + 2*y*b   
      
    def phiMag(xx, yy, xi, yj, coeffij, betaij):
        return coeffij * ( - np.cos(betaij) * ( F0(xx-xi-res/2,yy-yj-res/2)
                                               -F0(xx-xi+res/2,yy-yj-res/2)
                                               -F0(xx-xi-res/2,yy-yj+res/2)
                                               +F0(xx-xi+res/2,yy-yj+res/2) )
                           + np.sin(betaij) * ( F0(yy-yj-res/2,xx-xi-res/2)
                                               -F0(yy-yj+res/2,xx-xi-res/2)
                                               -F0(yy-yj-res/2,xx-xi+res/2)
                                               +F0(yy-yj+res/2,xx-xi+res/2) ) )
    
    '''CREATE COORDINATE GRIDS'''
    x = np.linspace(res/2,x_dim*res-res/2,num=x_dim)
    y = np.linspace(res/2,y_dim*res-res/2,num=y_dim)
    xx, yy = np.meshgrid(x,y)
    
    phase = np.zeros((y_dim,x_dim))
    
    for j in range(y_dim):
        for i in range(x_dim):
            phase += phiMag(xx, yy, x[i], y[j], coeff[j][i], beta[j][i])
  
    return phase
	
	
def display(phase, res, title):
    '''Display the phasemap as a colormesh.
    Arguments:
        phase - the phasemap that should be displayed
        res   - the resolution of the phasemap
        title - the title of the plot
    Returns:
        None
        
    '''    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    
    plt.pcolormesh(phase, cmap='gray')

    ticks = ax.get_xticks()*res
    ax.set_xticklabels(ticks.astype(int))
    ticks = ax.get_yticks()*res
    ax.set_yticklabels(ticks.astype(int))

    ax.set_title(title)
    ax.set_xlabel('x-axis [nm]')
    ax.set_ylabel('y-axis [nm]')
    
    plt.colorbar()
    plt.show()