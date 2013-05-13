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


def fourier_space(mag_data, b_0=1, padding=0):
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
    z_dim, y_dim, x_dim = mag_data.dim
    z_mag, y_mag, x_mag = mag_data.magnitude  
    
    # TODO: interpolation (redefine xDim,yDim,zDim) thickness ramp
    
    x_pad = x_dim/2 * padding
    y_pad = y_dim/2 * padding
    x_mag_big = np.zeros(((1 + padding) * y_dim, (1 + padding) * x_dim))
    y_mag_big = np.zeros(((1 + padding) * y_dim, (1 + padding) * x_dim))
    # TODO: padding so that x_dim and y_dim = 2^n
    x_mag_big[y_pad : y_pad + y_dim, x_pad : x_pad + x_dim] = x_mag
    y_mag_big[y_pad : y_pad + y_dim, x_pad : x_pad + x_dim] = y_mag
    
    x_mag_fft = np.fft.fftshift(np.fft.rfft2(x_mag_big), axes=0)
    y_mag_fft = np.fft.fftshift(np.fft.rfft2(y_mag_big), axes=0)
    nyq = 1 / res  # nyquist frequency
    x = np.linspace(      0, nyq/2, x_mag_fft.shape[1])
    y = np.linspace( -nyq/2, nyq/2, x_mag_fft.shape[0]+1)[:-1]
    xx, yy = np.meshgrid(x, y)
                         
    phase_fft = (1j * res * b_0) / (2 * PHI_0) * ((x_mag_fft * yy - y_mag_fft * xx) 
                                           / (xx ** 2 + yy ** 2 + 1e-18))
    phase_big = np.fft.irfft2(np.fft.ifftshift(phase_fft, axes=0))
    
    phase = phase_big[y_pad : y_pad + y_dim, x_pad : x_pad + x_dim]
    
    # TODO: Electrostatic Component

    return phase
      
      
def phi_pixel(method, n, m, res, b_0):  # TODO: rename
    if method == 'slab':    
        def F_h(n, m):
            a = np.log(res**2 * (n**2 + m**2))
            b = np.arctan(n / m)
            return n*a - 2*n + 2*m*b            
        coeff = -b_0 * res**2 / ( 2 * PHI_0 )
        return coeff * 0.5 * ( F_h(n-0.5, m-0.5) - F_h(n+0.5, m-0.5) 
                              -F_h(n-0.5, m+0.5) + F_h(n+0.5, m+0.5) ) 
    elif method == 'disc':
        coeff = - b_0 * res**2 / ( 2 * PHI_0 )
        in_or_out = np.logical_not(np.logical_and(n == 0, m == 0))
        return coeff * m / (n**2 + m**2 + 1E-30) * in_or_out


def real_space(mag_data, method, b_0=1, jacobi=None):
    '''Calculate phasemap from magnetization data (real space approach).
    Arguments:
        mag_data - MagDataLLG object (from magneticimaging.dataloader) storing the filename, 
                   coordinates and magnetization in x, y and z
    Returns:
        the phasemap as a 2 dimensional array
        
    '''    
    # TODO: Expand docstring!  

    res  = mag_data.res
    z_dim, y_dim, x_dim = mag_data.dim
    z_mag, y_mag, x_mag = mag_data.magnitude
    
    # TODO: proper projection algorithm       
    x_mag, y_mag, z_mag = x_mag.mean(0), y_mag.mean(0), z_mag.mean(0)
    
    beta = np.arctan2(y_mag, x_mag)

    mag = np.hypot(x_mag, y_mag) 
                
    '''CREATE COORDINATE GRIDS'''
    x = np.linspace(0,(x_dim-1),num=x_dim)
    y = np.linspace(0,(y_dim-1),num=y_dim)
    xx, yy = np.meshgrid(x,y)
     
    x_big = np.linspace(-(x_dim-1), x_dim-1, num=2*x_dim-1)
    y_big = np.linspace(-(y_dim-1), y_dim-1, num=2*y_dim-1)
    xx_big, yy_big = np.meshgrid(x_big, y_big)
    
    phi_cos = phi_pixel(method, xx_big, yy_big, res, b_0)
    phi_sin = phi_pixel(method, yy_big, xx_big, res, b_0)
            
    def phi_mag(i, j):  # TODO: rename
        return (np.cos(beta[j,i])*phi_cos[y_dim-1-j:(2*y_dim-1)-j, x_dim-1-i:(2*x_dim-1)-i]
               -np.sin(beta[j,i])*phi_sin[y_dim-1-j:(2*y_dim-1)-j, x_dim-1-i:(2*x_dim-1)-i])
                                          
    def phi_mag_deriv(i, j):  # TODO: rename
        return -(np.sin(beta[j,i])*phi_cos[y_dim-1-j:(2*y_dim-1)-j, x_dim-1-i:(2*x_dim-1)-i]
                +np.cos(beta[j,i])*phi_sin[y_dim-1-j:(2*y_dim-1)-j, x_dim-1-i:(2*x_dim-1)-i])
                                           
    def phi_mag_fd(i, j, h):  # TODO: rename
        return ((np.cos(beta[j,i]+h) - np.cos(beta[j,i])) / h 
                      * phi_cos[y_dim-1-j:(2*y_dim-1)-j, x_dim-1-i:(2*x_dim-1)-i]
               -(np.sin(beta[j,i]+h) - np.sin(beta[j,i])) / h 
                      * phi_sin[y_dim-1-j:(2*y_dim-1)-j, x_dim-1-i:(2*x_dim-1)-i])
    
    '''CALCULATE THE PHASE'''
    phase = np.zeros((y_dim, x_dim))
    
    # TODO: only iterate over pixels that have a magn. > threshold (first >0)
    if jacobi is not None:
        jacobi_fd = jacobi.copy()
    h = 0.0001
    
    for j in range(y_dim):
        for i in range(x_dim):
            #if (mag[j,i] != 0 ):#or jacobi is not None): # TODO: same result with or without?
                phi_mag_cache = phi_mag(i, j)
                phase += mag[j,i] * phi_mag_cache
                if jacobi is not None:
                    jacobi[:,i+x_dim*j] = phi_mag_cache.reshape(-1)
                    jacobi[:,x_dim*y_dim+i+x_dim*j] = (mag[j,i]*phi_mag_deriv(i,j)).reshape(-1)
                    
                    jacobi_fd[:,i+x_dim*j] = phi_mag_cache.reshape(-1)
                    jacobi_fd[:,x_dim*y_dim+i+x_dim*j] = (mag[j,i]*phi_mag_fd(i,j,h)).reshape(-1)  
                    
                    
    
    if jacobi is not None:
        jacobi_diff = jacobi_fd - jacobi
        assert (np.abs(jacobi_diff) < 1.0E-8).all(), 'jacobi matrix is not the same'
    
    return phase
    

def phase_elec(mag_data, b_0=1, v_0=0, v_acc=30000):
    # TODO: Docstring

    # TODO: Delete    
#    import pdb; pdb.set_trace()
    
    res  = mag_data.res
    z_dim, y_dim, x_dim = mag_data.dim
    z_mag, y_mag, x_mag = mag_data.magnitude  
    
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