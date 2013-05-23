# -*- coding: utf-8 -*-
"""Create and display a phase map from magnetization data."""


import numpy as np


# Physical constants
PHI_0 = 2067.83    # magnetic flux in T*nm²
H_BAR = 6.626E-34  # Planck constant in J*s
M_E   = 9.109E-31  # electron mass in kg
Q_E   = 1.602E-19  # electron charge in C
C     = 2.998E8    # speed of light in m/s


def phase_mag_fourier(res, projection, b_0=1, padding=0):
    '''Calculate phasemap from magnetization data (Fourier space approach).
    Arguments:
        res        - the resolution of the grid (grid spacing) in nm
        projection - projection of a magnetic distribution (created with pyramid.projector)
        b_0        - magnetic induction corresponding to a magnetization Mo in T (default: 1)
        padding    - factor for zero padding, the default is 0 (no padding), for a factor of n the 
                     number of pixels is increase by (1+n)**2, padded zeros are cropped at the end
    Returns:
        the phasemap as a 2 dimensional array
    
    '''
    v_dim, u_dim = np.shape(projection[0])
    v_mag, u_mag = projection
    # Create zero padded matrices:
    u_pad = u_dim/2 * padding
    v_pad = v_dim/2 * padding
    u_mag_big = np.zeros(((1 + padding) * v_dim, (1 + padding) * u_dim))
    v_mag_big = np.zeros(((1 + padding) * v_dim, (1 + padding) * u_dim))
    u_mag_big[v_pad : v_pad + v_dim, u_pad : u_pad + u_dim] = u_mag
    v_mag_big[v_pad : v_pad + v_dim, u_pad : u_pad + u_dim] = v_mag
    # Fourier transform of the two components:
    u_mag_fft = np.fft.fftshift(np.fft.rfft2(u_mag_big), axes=0)
    v_mag_fft = np.fft.fftshift(np.fft.rfft2(v_mag_big), axes=0)
    # Calculate the Fourier transform of the phase:
    nyq = 1 / res  # nyquist frequency
    u = np.linspace(      0, nyq/2, u_mag_fft.shape[1])
    v = np.linspace( -nyq/2, nyq/2, u_mag_fft.shape[0]+1)[:-1]
    uu, vv = np.meshgrid(u, v)
    coeff = (1j * res * b_0) / (2 * PHI_0)              
    phase_fft = coeff * (u_mag_fft*vv - v_mag_fft*uu) / (uu**2 + vv**2 + 1e-30)
    # Transform to real space and revert padding:
    phase_big = np.fft.irfft2(np.fft.ifftshift(phase_fft, axes=0))
    phase = phase_big[v_pad : v_pad + v_dim, u_pad : u_pad + u_dim]
    return phase
      

def phase_mag_real(res, projection, method, b_0=1, jacobi=None):
    '''Calculate phasemap from magnetization data (real space approach).
    Arguments:
        res        - the resolution of the grid (grid spacing) in nm
        projection - projection of a magnetic distribution (created with pyramid.projector)
        method     - String, describing the method to use for the pixel field ('slab' or 'disc')
        b_0        - magnetic induction corresponding to a magnetization Mo in T (default: 1)
        jacobi     - matrix in which to save the jacobi matrix (default: None, faster computation)
    Returns:
        the phasemap as a 2 dimensional array
    
    '''
    def phi_lookup(method, n, m, res, b_0):
        if method == 'slab':    
            def F_h(n, m):
                a = np.log(res**2 * (n**2 + m**2))
                b = np.arctan(n / m)
                return n*a - 2*n + 2*m*b            
            return coeff * 0.5 * ( F_h(n-0.5, m-0.5) - F_h(n+0.5, m-0.5) 
                                  -F_h(n-0.5, m+0.5) + F_h(n+0.5, m+0.5) ) 
        elif method == 'disc':
            in_or_out = np.logical_not(np.logical_and(n == 0, m == 0))
            return coeff * m / (n**2 + m**2 + 1E-30) * in_or_out
#    def phi_mag(i, j):  # TODO: rename
#        return (np.cos(beta[j,i])*phi_u[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i]
#               -np.sin(beta[j,i])*phi_v[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i])
#                                          
#    def phi_mag_deriv(i, j):  # TODO: rename
#        return -(np.sin(beta[j,i])*phi_u[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i]
#                +np.cos(beta[j,i])*phi_v[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i])
    # Process input parameters:
    v_dim, u_dim = np.shape(projection[0])
    v_mag, u_mag = projection 
    coeff = -b_0 * res**2 / ( 2 * PHI_0 )            
    # Create lookup-tables for the phase of one pixel:
    u = np.linspace(-(u_dim-1), u_dim-1, num=2*u_dim-1)
    v = np.linspace(-(v_dim-1), v_dim-1, num=2*v_dim-1)
    uu, vv = np.meshgrid(u, v)
    phi_u = phi_lookup(method, uu, vv, res, b_0)
    phi_v = phi_lookup(method, vv, uu, res, b_0)    
    '''CALCULATE THE PHASE'''
    phase = np.zeros((v_dim, u_dim))
    for j in range(v_dim):  # TODO: only iterate over pixels that have a magn. > threshold (first >0)
        for i in range(u_dim):
            if (u_mag[j,i] != 0 and v_mag[j,i] != 0) or jacobi is not None: # TODO: same result with or without?
                phase_u = phi_u[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i]                
                phase_v = phi_v[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i]
                phase += u_mag[j,i]*phase_u - v_mag[j,i]*phase_v
                if jacobi is not None:
                    jacobi[:,i+u_dim*j]             =  phase_u.reshape(-1)
                    jacobi[:,u_dim*v_dim+i+u_dim*j] = -phase_v.reshape(-1)
    
    return phase
    
    
def phase_mag_real_ANGLE(res, projection, method, b_0=1, jacobi=None):  # TODO: Modify
    '''Calculate phasemap from magnetization data (real space approach).
    Arguments:
        res        - the resolution of the grid (grid spacing) in nm
        projection - projection of a magnetic distribution (created with pyramid.projector)
        method     - String, describing the method to use for the pixel field ('slab' or 'disc')
        b_0        - magnetic induction corresponding to a magnetization Mo in T (default: 1)
        jacobi     - matrix in which to save the jacobi matrix (default: None, faster computation)
    Returns:
        the phasemap as a 2 dimensional array
    
    '''
    def phi_lookup(method, n, m, res, b_0):
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
    
    v_dim, u_dim = np.shape(projection[0])
    v_mag, u_mag = projection
    
    beta = np.arctan2(v_mag, u_mag)
    mag = np.hypot(u_mag, v_mag) 
                
    '''CREATE COORDINATE GRIDS'''
    u = np.linspace(0,(u_dim-1),num=u_dim)
    v = np.linspace(0,(v_dim-1),num=v_dim)
    uu, vv = np.meshgrid(u,v)
     
    u_lookup = np.linspace(-(u_dim-1), u_dim-1, num=2*u_dim-1)
    v_lookup = np.linspace(-(v_dim-1), v_dim-1, num=2*v_dim-1)
    uu_lookup, vv_lookup = np.meshgrid(u_lookup, v_lookup)
    
    phi_cos = phi_lookup(method, uu_lookup, vv_lookup, res, b_0)
    phi_sin = phi_lookup(method, vv_lookup, uu_lookup, res, b_0)
            
    def phi_mag(i, j):  # TODO: rename
        return (np.cos(beta[j,i])*phi_cos[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i]
               -np.sin(beta[j,i])*phi_sin[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i])
                                          
    def phi_mag_deriv(i, j):  # TODO: rename
        return -(np.sin(beta[j,i])*phi_cos[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i]
                +np.cos(beta[j,i])*phi_sin[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i])
                                           
    def phi_mag_fd(i, j, h):  # TODO: rename
        return ((np.cos(beta[j,i]+h) - np.cos(beta[j,i])) / h 
                      * phi_cos[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i]
               -(np.sin(beta[j,i]+h) - np.sin(beta[j,i])) / h 
                      * phi_sin[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i])
    
    '''CALCULATE THE PHASE'''
    phase = np.zeros((v_dim, u_dim))
    
    # TODO: only iterate over pixels that have a magn. > threshold (first >0)
    if jacobi is not None:
        jacobi_fd = jacobi.copy()
    h = 0.0001
    
    for j in range(v_dim):
        for i in range(u_dim):
            #if (mag[j,i] != 0 ):#or jacobi is not None): # TODO: same result with or without?
                phi_mag_cache = phi_mag(i, j)
                phase += mag[j,i] * phi_mag_cache
                if jacobi is not None:
                    jacobi[:,i+u_dim*j] = phi_mag_cache.reshape(-1)
                    jacobi[:,u_dim*v_dim+i+u_dim*j] = (mag[j,i]*phi_mag_deriv(i,j)).reshape(-1)
                    
                    jacobi_fd[:,i+u_dim*j] = phi_mag_cache.reshape(-1)
                    jacobi_fd[:,u_dim*v_dim+i+u_dim*j] = (mag[j,i]*phi_mag_fd(i,j,h)).reshape(-1)  
                    
                    
    
    if jacobi is not None:
        jacobi_diff = jacobi_fd - jacobi
        assert (np.abs(jacobi_diff) < 1.0E-8).all(), 'jacobi matrix is not the same'
    
    return phase
    

#def phase_elec(mag_data, v_0=0, v_acc=30000):
#    # TODO: Docstring
#
#    res  = mag_data.res
#    z_dim, y_dim, x_dim = mag_data.dim
#    z_mag, y_mag, x_mag = mag_data.magnitude  
#    
#    phase = np.logical_or(x_mag, y_mag, z_mag)    
#    
#    lam = H_BAR / np.sqrt(2 * M_E * v_acc * (1 + Q_E*v_acc / (2*M_E*C**2)))
#    
#    Ce = (2*pi*Q_E/lam * (Q_E*v_acc +   M_E*C**2) /
#            (Q_E*v_acc * (Q_E*v_acc + 2*M_E*C**2)))
#    
#    phase *= res * v_0 * Ce
#    
#    return phase