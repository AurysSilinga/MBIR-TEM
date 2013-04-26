# -*- coding: utf-8 -*-
"""Create and display a phase map from magnetization data."""


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi


PHI_0 = 2067.83    # magnetic flux in T*nm²
H_BAR = 6.626E-34  # TODO: unit
M_E   = 9.109E-31  # TODO: unit
Q_E   = 1.602E-19  # TODO: unit
C     = 2.998E8    # TODO: unit


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
    
    
def real_space_slab(mag_data, b_0=1, v_0=0, v_acc=30000):
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

    mag = np.hypot(x_mag, y_mag)    
     
    coeff = - b_0 * res**2 / ( 2 * PHI_0 ) / res # TODO: why / res

    def F_h(n, m):
        a = np.log(res**2 * (n**2 + m**2))
        b = np.arctan(n / m)
        return n*a - 2*n + 2*m*b
     
    def phi_pixel(n, m):  # TODO: rename
        return coeff * 0.5 * ( F_h(n-0.5, m-0.5) - F_h(n+0.5, m-0.5)
                              -F_h(n-0.5, m+0.5) + F_h(n+0.5, m+0.5) )
                
    def phi_mag(i, j):  # TODO: rename
        return mag[j,i]*(np.cos(beta[j,i])*phi_cos[y_dim-1-j:(2*y_dim-1)-j, 
                                                   x_dim-1-i:(2*x_dim-1)-i]
                        -np.sin(beta[j,i])*phi_sin[y_dim-1-j:(2*y_dim-1)-j,
                                                   x_dim-1-i:(2*x_dim-1)-i])
    
    '''CREATE COORDINATE GRIDS'''
    x = np.linspace(0,(x_dim-1),num=x_dim)
    y = np.linspace(0,(y_dim-1),num=y_dim)
    xx, yy = np.meshgrid(x,y)
     
    x_big = np.linspace(-(x_dim-1), x_dim-1, num=2*x_dim-1)
    y_big = np.linspace(-(y_dim-1), y_dim-1, num=2*y_dim-1)
    xx_big, yy_big = np.meshgrid(x_big, y_big)
    
    phi_cos = phi_pixel(xx_big, yy_big)
    phi_sin = phi_pixel(yy_big, xx_big)
    
    display_phase(phi_cos, res, 'Phase of one Pixel (Cos - Part)')
#    display_phase(phi_sin, res, 'Phase of one Pixel (Sin - Part)')
    
    '''CALCULATE THE PHASE'''
    phase = np.zeros((y_dim, x_dim))
    
    # TODO: only iterate over pixels that have a magn. > threshold (first >0)
    for j in range(y_dim):
        for i in range(x_dim):
            phase += phi_mag(i, j)
    
    return (phase, phi_cos)
    
    
def real_space_disc(mag_data, b_0=1, v_0=0, v_acc=30000):
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

    mag = np.hypot(x_mag, y_mag)
    
    coeff = - b_0 * res**2 / ( 2 * PHI_0 ) / res # TODO: why / res

    def phi_pixel(n, m):
        in_or_out = np.logical_not(np.logical_and(n == 0, m == 0))
        result = coeff * m / (n**2 + m**2 + 1E-30) * in_or_out
        return result
#        r = np.hypot(n, m)
#        r[y_dim-1,x_dim-1] = 1E-18  # Prevent div zero error at disc center
#        return coeff / r**2 * m * (r != 0) # (r > R) = 0 for disc center

    def phi_mag(i, j):
        return mag[j,i]*(np.cos(beta[j,i])*phi_cos[y_dim-1-j:(2*y_dim-1)-j, 
                                                   x_dim-1-i:(2*x_dim-1)-i]
                        -np.sin(beta[j,i])*phi_sin[y_dim-1-j:(2*y_dim-1)-j,
                                                   x_dim-1-i:(2*x_dim-1)-i])
    
    '''CREATE COORDINATE GRIDS'''
    x = np.linspace(0,(x_dim-1),num=x_dim)
    y = np.linspace(0,(y_dim-1),num=y_dim)
    xx, yy = np.meshgrid(x,y)
     
    x_big = np.linspace(-(x_dim-1), x_dim-1, num=2*x_dim-1)
    y_big = np.linspace(-(y_dim-1), y_dim-1, num=2*y_dim-1)
    xx_big, yy_big = np.meshgrid(x_big, y_big)
    
    phi_cos = phi_pixel(xx_big, yy_big)
    phi_sin = phi_pixel(yy_big, xx_big)
    
    display_phase(phi_cos, res, 'Phase of one Pixel (Cos - Part)')
#    display_phase(phi_sin, res, 'Phase of one Pixel (Sin - Part)')
    
    '''CALCULATE THE PHASE'''
    phase = np.zeros((y_dim, x_dim))
    
    # TODO: only iterate over pixels that have a magn. > threshold (first >0)
    for j in range(y_dim):
        for i in range(x_dim):
            phase += phi_mag(i, j)
    
    return (phase, phi_cos)
    

def phase_elec(mag_data, b_0=1, v_0=0, v_acc=30000):
    # TODO: Docstring

    # TODO: Delete    
#    import pdb; pdb.set_trace()
    
    res  = mag_data.res
    x_dim, y_dim, z_dim = mag_data.dim
    x_mag, y_mag, z_mag = mag_data.magnitude 
    
    phase = np.logical_or(x_mag, y_mag, z_mag)    
    
    lam = H_BAR / np.sqrt(2 * M_E * v_acc * (1 + Q_E*v_acc / (2*M_E*C**2)))
    
    Ce = (2*pi*Q_E/lam * (Q_E*v_acc +   M_E*C**2) /
            (Q_E*v_acc * (Q_E*v_acc + 2*M_E*C**2)))
    
    phase *= res * v_0 * Ce
    
    return phase
    
	
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
    axis.set_xticklabels(ticks)
    ticks = axis.get_yticks()*res
    axis.set_yticklabels(ticks)

    axis.set_title(title)
    axis.set_xlabel('x-axis [nm]')
    axis.set_ylabel('y-axis [nm]')
    
    fig = plt.gcf()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.show()