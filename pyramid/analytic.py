# -*- coding: utf-8 -*-
"""Create simple phasemaps from analytic solutions."""


import numpy as np
from numpy import pi


PHI_0 = 2067.83  # magnetic flux in T*nm²


def phase_mag_slab(dim, res, beta, center, width, b_0):
    '''INPUT VARIABLES'''
    z_dim, y_dim, x_dim = dim
    # y0, x0 have to be in the center of a pixel, hence: cellindex + 0.5
    y0 = res * (center[1] + 0.5)
    x0 = res * (center[2] + 0.5)
    # Ly, Lx have to be odd, because the slab borders should not lie in the
    # center of a pixel (so L/2 can't be an integer)
    Ly = res * ( width[1] + (width[1]+1)%2) 
    Lx = res * ( width[2] + (width[2]+1)%2)   
    
    '''COMPUTATION MAGNETIC PHASE SHIFT (REAL SPACE) SLAB'''

    coeff = b_0 * res / ( 4 * PHI_0 )
     
    def F0(x,y):
        a = np.log(x**2 + y**2)
        b = np.arctan(x / y)
        return x*a - 2*x + 2*y*b   
    
    def phiMag(x,y):
        return coeff * ( - np.cos(beta) * ( F0(x-x0-Lx/2, y-y0-Ly/2)
                                           -F0(x-x0+Lx/2, y-y0-Ly/2)
                                           -F0(x-x0-Lx/2, y-y0+Ly/2)
                                           +F0(x-x0+Lx/2, y-y0+Ly/2) )
                         + np.sin(beta) * ( F0(y-y0-Ly/2, x-x0-Lx/2)
                                           -F0(y-y0+Ly/2, x-x0-Lx/2)
                                           -F0(y-y0-Ly/2, x-x0+Lx/2)
                                           +F0(y-y0+Ly/2, x-x0+Lx/2) ) )
             
    '''CREATE COORDINATE GRIDS'''
    x = np.linspace(res/2,x_dim*res-res/2,num=x_dim)
    y = np.linspace(res/2,y_dim*res-res/2,num=y_dim)
    xx, yy = np.meshgrid(x,y)
    
    return phiMag(xx, yy)
    
    
def phase_mag_disc(dim, res, beta, center, radius, b_0):
    '''INPUT VARIABLES'''
    z_dim, y_dim, x_dim = dim
    # y0, x0 have to be in the center of a pixel, hence: cellindex + 0.5
    y0 = res * (center[1] + 0.5)
    x0 = res * (center[2] + 0.5)
    # TODO: Explanation
    # Ly, Lx have to be odd, because the slab borders should not lie in the
    # center of a pixel (so L/2 can't be an integer)
    R  = res * radius
    
    '''COMPUTATION MAGNETIC PHASE SHIFT (REAL SPACE) DISC'''
      
    coeff = - pi * res * b_0 / ( 2 * PHI_0 )
    
#    import pdb; pdb.set_trace()    
    
    def phiMag(x,y):
        r = np.hypot(x-x0, y-y0)
        r[center[1], center[2]] = 1E-30
        result = coeff * ((y-y0) * np.cos(beta) - (x-x0) * np.sin(beta))
        in_or_out = 1 * (r < R) + (R / r) ** 2 * (r > R)
#        import pdb; pdb.set_trace() 
        result *= in_or_out
        return result
    
    '''CREATE COORDINATE GRIDS'''
    x = np.linspace(res/2,x_dim*res-res/2,num=x_dim)
    y = np.linspace(res/2,y_dim*res-res/2,num=y_dim)
    xx, yy = np.meshgrid(x,y)
    
    return phiMag(xx, yy)
    
    
def phase_mag_sphere(dim, res, beta, center, radius, b_0):
    
    # TODO: Sphere is equivalent to disc, if only one pixel in z is used!
    
    '''INPUT VARIABLES'''
    y_dim, x_dim = dim
    y0, x0 = center
    R = radius
    
    '''COMPUTATION MAGNETIC PHASE SHIFT (REAL SPACE) SPHERE''' 
      
    coeff = - 2/3 * pi * R**3 * b_0 / PHI_0 * res/R
      
    def phiMag(x,y):
        r = np.sqrt((x-x0) ** 2 + (y-y0) ** 2)      
        result = coeff  / r**2 * ((y-y0) * np.cos(beta) - (x-x0) * np.sin(beta))
        in_or_out = 1 * (r > R) + (1 - (1-(r/R)**2)**(3/2)) * (r < R)    
        result *= in_or_out
        return result
    
    '''CREATE COORDINATE GRIDS'''
    x = np.linspace(res/2,x_dim*res-res/2,num=x_dim)
    y = np.linspace(res/2,y_dim*res-res/2,num=y_dim)
    xx, yy = np.meshgrid(x,y)
    
    return phiMag(xx, yy)