# -*- coding: utf-8 -*-
"""Create simple LLG Files which describe magnetization in 2D (z-Dim=1)."""


import numpy as np
from magdata import MagData


def shape_slab(dim, center, width):
    '''Get the magnetic shape of a slab.
    Arguments:
        dim    - the dimensions of the grid, shape(y, x)
        center - center of the slab in pixel coordinates, shape(y, x)
        width  - width of the slab in pixel coordinates, shape(y, x)
    Returns:
        the magnetic shape as a 2D-boolean-array.
        
    '''
    mag_shape = np.array([[[abs(x - center[2]) <= width[2] / 2
                        and abs(y - center[1]) <= width[1] / 2
                        and abs(z - center[0]) <= width[0] / 2
                        for x in range(dim[2])] for y in range(dim[1])] for z in range(dim[0])])
    return mag_shape


def shape_disc(dim, center, radius, height):
    '''Get the magnetic shape of a disc.
    Arguments:
        dim    - the dimensions of the grid, shape(y, x)
        center - center of the disc in pixel coordinates, shape(y, x)
        radius - radius of the disc in pixel coordinates, shape(y, x)
    Returns:
        the magnetic shape as a 2D-boolean-array.
        
    '''# TODO: up till now only in z-direction
    mag_shape = np.array([[[np.hypot(x - center[2], y - center[1]) <= radius
                        and abs(z - center[0]) <= height / 2
                        for x in range(dim[2])] for y in range(dim[1])] for z in range(dim[0])])    
    return mag_shape

    
def shape_filament(dim, pos, x_or_y):
    '''Get the magnetic shape of a single filament in x or y direction.
    Arguments:
        dim    - the dimensions of the grid, shape(y, x)
        pos    - position of the filament (pixel coordinate)
        x_or_y - string that determines the orientation of the filament
                 ('y' or 'x')
    Returns:
        the magnetic shape as a 2D-boolean-array.
        
    '''# TODO: up till now no z-direction
    mag_shape = np.zeros(dim)
    if x_or_y == 'y':
        mag_shape[:, :, pos] = 1
    else:
        mag_shape[:, pos, :] = 1
    return mag_shape


def shape_alternating_filaments(dim, spacing, x_or_y):
    '''Get the magnetic shape of alternating filaments in x or y direction.
    Arguments:
        dim     - the dimensions of the grid, shape(y, x)
        spacing - the distance between two filaments (pixel coordinate)
        x_or_y  - string that determines the orientation of the filament
                  ('y' or 'x')
    Returns:
        the magnetic shape as a 2D-boolean-array.
        
    '''
    mag_shape = np.zeros(dim)
    if x_or_y == 'y':
        for i in range(0, dim[1], spacing):
            mag_shape[:, :, i] = 1 - 2 * (int(i / spacing) % 2)  # 1 or -1
    else:
        for i in range(0, dim[0], spacing):
            mag_shape[:, i, :] = 1 - 2 * (int(i / spacing) % 2)  # 1 or -1            
    return mag_shape

    
def shape_single_pixel(dim, pixel):
    '''Get the magnetic shape of a single magnetized pixel.
    Arguments:
        dim   - the dimensions of the grid, shape(y, x)
        pixel - the coordinates of the magnetized pixel, shape(y, x)
    Returns:
        the magnetic shape as a 2D-boolean-array.
        
    '''
    mag_shape = np.zeros(dim)
    mag_shape[pixel] = 1
    return mag_shape


def hom_mag(dim, res, mag_shape, beta, magnitude=1):
    '''Create homog. magnetization data, saved in a file with LLG-format.
    Arguments:
        dim       - the dimensions of the grid, shape(y, x)
        res       - the resolution of the grid 
                    (real space distance between two points)
        beta      - the angle of the magnetization
        filename  - the name of the file in which to save the data
        mag_shape - an array of shape dim, representing the shape of the magnetic object,
                    a few are supplied in this module
    Returns:
        the the magnetic distribution as a 2D-boolean-array.
        
    ''' # TODO: renew Docstring
    z_mag = np.array(np.zeros(dim))        
    y_mag = np.array(np.ones(dim)) * np.sin(beta) * mag_shape * magnitude
    x_mag = np.array(np.ones(dim)) * np.cos(beta) * mag_shape * magnitude
    return z_mag, y_mag, x_mag
    
                                      
def create_mag_dist(dim, res, mag_shape_list, beta_list, magnitude_list=None):
    # TODO: Docstring: can take just one or a list of objects!  OR JUST A LIST!!!  
    # If no relative magnitude is specified, 1 is assumed for every homog. object:
    if magnitude_list is None:
        magnitude_list = np.ones(np.size(beta_list))    
    # For every shape of a homogeneous object a relative magnitude and angle have to be set:
    assert (np.shape(mag_shape_list)[0],) == np.shape(beta_list) == np.shape(magnitude_list), \
        'Lists have not the same length!'
    x_mag = np.zeros(dim)
    y_mag = np.zeros(dim)
    z_mag = np.zeros(dim)    
    for i in range(np.size(beta_list)):
        pixel_mag = hom_mag(dim, res, mag_shape_list[i], beta_list[i], magnitude_list[i])    
        z_mag += pixel_mag[0]
        y_mag += pixel_mag[1]
        x_mag += pixel_mag[2]
    return MagData(res, z_mag, y_mag, x_mag)