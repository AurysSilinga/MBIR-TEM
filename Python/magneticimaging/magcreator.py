# -*- coding: utf-8 -*-
"""Create simple LLG Files which describe magnetization in 2D (z-Dim=1)."""


import numpy as np
import matplotlib.pyplot as plt


def slab(dim, params):
    '''Get the magnetic shape of a slab.
    Arguments:
        dim    - the dimensions of the grid, shape(y, x)
        center - center of the slab in pixel coordinates, shape(y, x)
        width  - width of the slab in pixel coordinates, shape(y, x)
    Returns:
        the magnetic shape as a 2D-boolean-array.
        
    '''
    center, width = params
    shape_mag = np.array([[abs(x - center[1]) <= width[1] / 2
                       and abs(y - center[0]) <= width[0] / 2
                       for x in range(dim[1])] for y in range(dim[0])])
    return shape_mag


def disc(dim, params):
    '''Get the magnetic shape of a disc.
    Arguments:
        dim    - the dimensions of the grid, shape(y, x)
        center - center of the disc in pixel coordinates, shape(y, x)
        radius - radius of the disc in pixel coordinates, shape(y, x)
    Returns:
        the magnetic shape as a 2D-boolean-array.
        
    '''
    center, radius = params
    shape_mag = np.array([[(np.sqrt((x - center[1]) ** 2 +
                                    (y - center[0]) ** 2) <= radius)
                                    for x in range(dim[1])] 
                                    for y in range(dim[0])])    
    return shape_mag

    
def filament(dim, params):
    '''Get the magnetic shape of a single filament in x or y direction.
    Arguments:
        dim    - the dimensions of the grid, shape(y, x)
        pos    - position of the filament (pixel coordinate)
        x_or_y - string that determines the orientation of the filament
                 ('y' or 'x')
    Returns:
        the magnetic shape as a 2D-boolean-array.
        
    '''
    pos, x_or_y = params
    shape_mag = np.zeros(dim)
    if x_or_y == 'y':
        shape_mag[:, pos] = 1
    else:
        shape_mag[pos, :] = 1
    return shape_mag


def alternating_filaments(dim, params):
    '''Get the magnetic shape of alternating filaments in x or y direction.
    Arguments:
        dim     - the dimensions of the grid, shape(y, x)
        spacing - the distance between two filaments (pixel coordinate)
        x_or_y  - string that determines the orientation of the filament
                  ('y' or 'x')
    Returns:
        the magnetic shape as a 2D-boolean-array.
        
    '''
    spacing, x_or_y = params
    shape_mag = np.zeros(dim)
    if x_or_y == 'y':
        # TODO: List comprehension:
        for i in range(0, dim[1], spacing):
            shape_mag[:, i] = 1 - 2 * (int(i / spacing) % 2)  # 1 or -1
    else:
        for i in range(0, dim[0], spacing):
            shape_mag[i, :] = 1 - 2 * (int(i / spacing) % 2)  # 1 or -1            
    return shape_mag

    
def single_pixel(dim, params):
    '''Get the magnetic shape of a single magnetized pixel.
    Arguments:
        dim   - the dimensions of the grid, shape(y, x)
        pixel - the coordinates of the magnetized pixel, shape(y, x)
    Returns:
        the magnetic shape as a 2D-boolean-array.
        
    '''
    pixel = params
    shape_mag = np.zeros(dim)
    shape_mag[pixel[0], pixel[1]] = 1
    return shape_mag


def create_hom_mag(dim, res, beta, shape_fun, params,
                   filename='output.txt', plot_mag_distr=False):
    '''Create homog. magnetization data, saved in a file with LLG-format.
    Arguments:
        dim       - the dimensions of the grid, shape(y, x)
        res       - the resolution of the grid 
                    (real space distance between two points)
        beta      - the angle of the magnetization
        filename  - the name of the file in which to save the data
        shape_fun - the function to determine the shape of the magnetization,
                    several are defined in this module
        params   - a tuple of parameters used in shape_fun
    Returns:
        the the magnetic shape as a 2D-boolean-array.
        
    '''
    res *= 1.0E-9 / 1.0E-2  # from nm to cm     
    
    x = np.linspace(res / 2, dim[1] * res - res / 2, num=dim[1])
    y = np.linspace(res / 2, dim[0] * res - res / 2, num=dim[0])
    xx, yy = np.meshgrid(x, y)
    
    shape_mag = shape_fun(dim, params)                        
                        
    x_mag = np.array(np.ones(dim)) * np.cos(beta) * shape_mag
    y_mag = np.array(np.ones(dim)) * np.sin(beta) * shape_mag
    z_mag = np.array(np.zeros(dim))
    
    if (plot_mag_distr):
        fig = plt.figure()
        fig.add_subplot(111, aspect='equal')
        plt.quiver(x_mag, y_mag, pivot='middle', angles='xy', scale_units='xy', 
                   scale=1, headwidth=6, headlength=7)    
    
    xx = np.reshape(xx,(-1))
    yy = np.reshape(yy,(-1))
    zz = np.array(np.ones(dim[0] * dim[1]) * res / 2)
    x_mag   = np.reshape(x_mag,(-1))
    y_mag   = np.reshape(y_mag,(-1))
    z_mag   = np.array(np.zeros(dim[0] * dim[1]))
           
    data = np.array([xx, yy, zz, x_mag, y_mag, z_mag]).T
    with open(filename,'w') as mag_file:
        mag_file.write('LLGFileCreator2D: %s\n' % filename.replace('.txt', ''))
        mag_file.write('    %d    %d    %d\n' % (dim[1], dim[0], 1))
        mag_file.writelines('\n'.join('   '.join('{:7.6e}'.format(cell) 
                                      for cell in row) for row in data) )