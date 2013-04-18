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

    # TODO: Delete
#    import pdb; pdb.set_trace()    
    
    res  = mag_data.res
    x_dim, y_dim, z_dim = mag_data.dim
    x_mag, y_mag, z_mag = mag_data.magnitude    
    
    beta = np.arctan2(y_mag, x_mag)

    abs_mag = np.sqrt(x_mag**2 + y_mag**2)    
     
    coeff = abs_mag * res / ( 4 * PHI_0 ) / res # TODO: Lz = res 

    def F0(n, m):
        a = np.log(res**2 * (n**2 + m**2))
        b = np.arctan(n / m)
        return res * (n*a - 2*n + 2*m*b)
     
    def F_part(n, m):
        return ( F0(n-0.5, m-0.5) - F0(n+0.5, m-0.5)
                -F0(n-0.5, m+0.5) + F0(n+0.5, m+0.5) )
    
    def phiMag(xx, yy, xi, yj, coeffij, betaij):
        return coeffij * ( - np.cos(betaij) * F_part(xx-xi, yy-yj)
                           + np.sin(betaij) * F_part(yy-yj, xx-xi) )
    
    '''CREATE COORDINATE GRIDS'''
    x = np.linspace(0,(x_dim-1),num=x_dim)
    y = np.linspace(0,(y_dim-1),num=y_dim)
    xx, yy = np.meshgrid(x,y)
    
    xF = np.linspace(-(x_dim-1), x_dim-1, num=2*x_dim-1)
    yF = np.linspace(-(y_dim-1), y_dim-1, num=2*y_dim-1)
    xxF, yyF = np.meshgrid(xF,yF)
    
    F_part_cos = F_part(xxF, yyF)
    F_part_sin = F_part(yyF, xxF)
    
    display_phase(F_part_cos, res, 'F_part_cos')
    display_phase(F_part_sin, res, 'F_part_sin')      
    
    phase = np.zeros((y_dim,x_dim))
    
    for j in y:
        for i in x:
            phase += phiMag(xx, yy, i, j, coeff[j,i], beta[j,i])
            
    return phase
    
    
    xF = np.linspace(-(x_dim-1), x_dim-1, num=2*x_dim-1)
    yF = np.linspace(-(y_dim-1), y_dim-1, num=2*y_dim-1)
    xxF, yyF = np.meshgrid(xF,yF)
    
    F_part_cos = F_part(xxF, yyF)
    F_part_sin = F_part(yyF, xxF)
    
    
    display_phase(F_part_cos, res, 'F_part_cos')
    display_phase(F_part_sin, res, 'F_part_sin')    
    
    def phiMag2(xx, yy, i, j):
        #x_ind = xxF[yy]
        
        return coeff[j,i] * ( - np.cos(beta[j,i]) * F_part_cos[yy.min()-j:(2*yy.max()-1)-j, xx.min()-i:(2*xx.max()-1)-i]
                              + np.sin(beta[j,i]) * F_part_sin[xx.min()-i:(2*xx.max()-1)-i, yy.min()-j:(2*yy.max()-1)-j] )
    
    
    phase2 = np.zeros((y_dim*x_dim, y_dim, x_dim))
    
    for j in range(y_dim):
        for i in range(x_dim):
            phase2 += phiMag2(xx, yy, 0, 0)
                
   
    
    phase2 = np.sum(phase2, axis=0)
    
    return phase2
    
	
def display_phase(phase, res, title, axis=None):
    '''Display the phasemap as a colormesh.
    Arguments:
        phase - the phasemap that should be displayed
        res   - the resolution of the phasemap
        title - the title of the plot
    Returns:
        None
        
    '''
    if axis == None:
        fig = plt.figure()
        axis = fig.add_subplot(1,1,1, aspect='equal')
    
    im = plt.pcolormesh(phase, cmap='gray')

    ticks = axis.get_xticks()*res
    axis.set_xticklabels(ticks.astype(int))
    ticks = axis.get_yticks()*res
    axis.set_yticklabels(ticks.astype(int))

    axis.set_title(title)
    axis.set_xlabel('x-axis [nm]')
    axis.set_ylabel('y-axis [nm]')
    
    fig = plt.gcf()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.show()