# -*- coding: utf-8 -*-
"""Create simple phasemaps from analytic solutions."""

# TODO: Currently just for projections along the z-axis!


import numpy as np
from numpy import pi


PHI_0 = -2067.83  # magnetic flux in T*nm²


def phase_mag_slab(dim, res, phi, center, width, b_0=1):
    '''Get the analytic solution for a phase map of a slab of specified dimensions.
    Arguments:
        dim    - the dimensions of the grid, shape(z, y, x)
        res    - the resolution of the grid (grid spacing) in nm
        phi    - the azimuthal angle, describing the direction of the magnetized object
        center - the center of the slab in pixel coordinates, shape(z, y, x)
        width  - the width of the slab in pixel coordinates, shape(z, y, x)
        b_0    - magnetic induction corresponding to a magnetization Mo in T (default: 1)
    Returns:
        the analytic solution for the phase map

    '''
    # Function for the phase:
    def phiMag(x,  y):
        def F0(x, y):
            a = np.log(x**2 + y**2 + 1E-30)
            b = np.arctan(x / (y+1E-30))
            return x*a - 2*x + 2*y*b
        return coeff * Lz * (- np.cos(phi) * (F0(x-x0-Lx/2, y-y0-Ly/2)
                                             - F0(x-x0+Lx/2, y-y0-Ly/2)
                                             - F0(x-x0-Lx/2, y-y0+Ly/2)
                                             + F0(x-x0+Lx/2, y-y0+Ly/2))
                             + np.sin(phi) * (F0(y-y0-Ly/2, x-x0-Lx/2)
                                             - F0(y-y0+Ly/2, x-x0-Lx/2)
                                             - F0(y-y0-Ly/2, x-x0+Lx/2)
                                             + F0(y-y0+Ly/2, x-x0+Lx/2)))
    # Process input parameters:
    z_dim, y_dim, x_dim = dim
    y0 = res * (center[1] + 0.5)  # y0, x0 define the center of a pixel,
    x0 = res * (center[2] + 0.5)  # hence: (cellindex + 0.5) * resolution
    Lz, Ly, Lx = res * width[0], res * width[1], res * width[2]
    coeff = b_0 / (4*PHI_0)
    # Create grid:
    x = np.linspace(res/2, x_dim*res-res/2, num=x_dim)
    y = np.linspace(res/2, y_dim*res-res/2, num=y_dim)
    xx, yy = np.meshgrid(x, y)
    # Return phase:
    return phiMag(xx, yy)


def phase_mag_disc(dim, res, phi, center, radius, height, b_0=1):
    '''Get the analytic solution for a phase map of a disc of specified dimensions.
    Arguments:
        dim    - the dimensions of the grid, shape(z, y, x)
        res    - the resolution of the grid (grid spacing) in nm
        phi    - the azimuthal angle, describing the direction of the magnetized object
        center - the center of the disc in pixel coordinates, shape(z, y, x)
        radius - the radius of the disc in pixel coordinates (scalar value)
        height - the height of the disc in pixel coordinates (scalar value)
        b_0    - magnetic induction corresponding to a magnetization Mo in T (default: 1)
    Returns:
        the analytic solution for the phase map

    '''
    # Function for the phase:
    def phiMag(x, y):
        r = np.hypot(x - x0, y - y0)
        r[center[1], center[2]] = 1E-30
        result = coeff * Lz * ((y - y0) * np.cos(phi) - (x - x0) * np.sin(phi))
        result *= np.where(r <= R, 1, (R / r) ** 2)
        return result
    # Process input parameters:
    z_dim, y_dim, x_dim = dim
    y0 = res * (center[1] + 0.5)  # y0, x0 have to be in the center of a pixel,
    x0 = res * (center[2] + 0.5)  # hence: cellindex + 0.5
    Lz = res * height
    R = res * radius
    coeff = - pi * b_0 / (2*PHI_0)
    # Create grid:
    x = np.linspace(res/2, x_dim*res-res/2, num=x_dim)
    y = np.linspace(res/2, y_dim*res-res/2, num=y_dim)
    xx, yy = np.meshgrid(x, y)
    # Return phase:
    return phiMag(xx, yy)


def phase_mag_sphere(dim, res, phi, center, radius, b_0=1):
    '''Get the analytic solution for a phase map of a sphere of specified dimensions.
    Arguments:
        dim    - the dimensions of the grid, shape(z, y, x)
        res    - the resolution of the grid (grid spacing) in nm
        phi    - the azimuthal angle, describing the direction of the magnetized object
        center - the center of the sphere in pixel coordinates, shape(z, y, x)
        radius - the radius of the sphere in pixel coordinates (scalar value)
        b_0    - magnetic induction corresponding to a magnetization Mo in T (default: 1)
    Returns:
        the analytic solution for the phase map

    '''
    # Function for the phase:
    def phiMag(x, y):
        r = np.hypot(x - x0, y - y0)
        r[center[1], center[2]] = 1E-30
        result = coeff * R ** 3 / r ** 2 * ((y - y0) * np.cos(phi) - (x - x0) * np.sin(phi))
        result *= np.where(r > R, 1, (1 - (1 - (r / R) ** 2) ** (3. / 2.)))
        return result
    # Process input parameters:
    z_dim, y_dim, x_dim = dim
    y0 = res * (center[1] + 0.5)  # y0, x0 have to be in the center of a pixel,
    x0 = res * (center[2] + 0.5)  # hence: cellindex + 0.5
    R = res * radius
    coeff = - 2./3. * pi * b_0 / PHI_0
    # Create grid:
    x = np.linspace(res / 2, x_dim * res - res / 2, num=x_dim)
    y = np.linspace(res / 2, y_dim * res - res / 2, num=y_dim)
    xx, yy = np.meshgrid(x, y)
    # Return phase:
    return phiMag(xx, yy)


def phase_mag_vortex(dim, res, center, radius, height, b_0=1):
    '''Get the analytic solution for a phase map of a sharp vortex disc of specified dimensions.
    Arguments:
        dim    - the dimensions of the grid, shape(z, y, x)
        res    - the resolution of the grid (grid spacing) in nm
        center - the center of the disc in pixel coordinates, shape(z, y, x)
        radius - the radius of the disc in pixel coordinates (scalar value)
        height - the height of the disc in pixel coordinates (scalar value)
        b_0    - magnetic induction corresponding to a magnetization Mo in T (default: 1)
    Returns:
        the analytic solution for the phase map

    '''
    # Function for the phase:
    def phiMag(x, y):
        r = np.hypot(x - x0, y - y0)
        result = coeff * np.where(r <= R, r - R, 0)
        return result
    # Process input parameters:
    z_dim, y_dim, x_dim = dim
    y0 = res * (center[1] + 0.5)  # y0, x0 have to be in the center of a pixel,
    x0 = res * (center[2] + 0.5)  # hence: cellindex + 0.5
    Lz = res * height
    R = res * radius
    coeff = pi * b_0 * Lz / PHI_0
    # Create grid:
    x = np.linspace(res/2, x_dim*res-res/2, num=x_dim)
    y = np.linspace(res/2, y_dim*res-res/2, num=y_dim)
    xx, yy = np.meshgrid(x, y)
    # Return phase:
    return phiMag(xx, yy)
