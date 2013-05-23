# -*- coding: utf-8 -*-
"""Create simple LLG Files which describe magnetization in 2D (z-Dim=1)."""


import numpy as np
from pyramid.magdata import MagData


class Shapes:
    
    '''Class containing functions for generating shapes.'''
       
    @classmethod
    def slab(cls, dim, center, width):
        '''Get the magnetic shape of a slab.
        Arguments:
            dim    - the dimensions of the grid, shape(z, y, x)
            center - the center of the slab in pixel coordinates, shape(z, y, x)
            width  - the width of the slab in pixel coordinates, shape(z, y, x)
        Returns:
            the magnetic shape as a 3D-array with ones and zeros
            
        '''
        assert np.shape(dim)    == (3,), 'Parameter dim has to be a shape of 3 dimensions!'
        assert np.shape(center) == (3,), 'Parameter center has to be a shape of 3 dimensions!'
        assert np.shape(width)  == (3,), 'Parameter width has to be a shape of 3 dimensions!'
        mag_shape = np.array([[[abs(x - center[2]) <= width[2] / 2
                            and abs(y - center[1]) <= width[1] / 2
                            and abs(z - center[0]) <= width[0] / 2
                            for x in range(dim[2])] 
                            for y in range(dim[1])] 
                            for z in range(dim[0])])
        return mag_shape
    
    @classmethod
    def disc(cls, dim, center, radius, height, axis='z'):
        '''Get the magnetic shape of a zylindrical disc in x-, y-, or z-direction.
        Arguments:
            dim    - the dimensions of the grid, shape(z, y, x)
            center - the center of the disc in pixel coordinates, shape(z, y, x)
            radius - the radius of the disc in pixel coordinates (scalar value)
            height - the height of the disc in pixel coordinates (scalar value)
            axis   - the orientation of the disc axis, (String: 'x', 'y', 'z'), default = 'z'
        Returns:
            the magnetic shape as a 3D-array with ones and zeros
            
        '''
        assert np.shape(dim)    == (3,), 'Parameter dim has to be a shape of 3 dimensions!'
        assert np.shape(center) == (3,), 'Parameter center has to be a shape of 3 dimensions!'
        assert radius > 0 and np.shape(radius) == (), 'Radius has to be positive scalar value!'
        assert height > 0 and np.shape(height) == (), 'Height has to be positive scalar value!'
        assert axis == 'z' or axis == 'y' or axis == 'x', 'Axis has to be x, y or z (as String)!'
        if axis == 'z':
            mag_shape = np.array([[[np.hypot(x - center[2], y - center[1]) <= radius
                                and abs(z - center[0]) <= height / 2
                                for x in range(dim[2])] 
                                for y in range(dim[1])] 
                                for z in range(dim[0])])
        elif axis == 'y':
            mag_shape = np.array([[[np.hypot(x - center[2], z - center[0]) <= radius
                                and abs(y - center[1]) <= height / 2
                                for x in range(dim[2])] 
                                for y in range(dim[1])] 
                                for z in range(dim[0])])
        elif axis == 'x':
            mag_shape = np.array([[[np.hypot(y - center[1], z - center[0]) <= radius
                                and abs(x - center[2]) <= height / 2
                                for x in range(dim[2])] 
                                for y in range(dim[1])] 
                                for z in range(dim[0])]) 
        return mag_shape
           
    @classmethod
    def sphere(cls, dim, center, radius):
        '''Get the magnetic shape of a sphere.
        Arguments:
            dim    - the dimensions of the grid, shape(z, y, x)
            center - the center of the disc in pixel coordinates, shape(z, y, x)
            radius - the radius of the disc in pixel coordinates (scalar value)
            height - the height of the disc in pixel coordinates (scalar value)
            axis   - the orientation of the disc axis, (String: 'x', 'y', 'z'), default = 'z'
        Returns:
            the magnetic shape as a 3D-array with ones and zeros
            
        '''
        assert np.shape(dim)    == (3,), 'Parameter dim has to be a shape of 3 dimensions!'
        assert np.shape(center) == (3,), 'Parameter center has to be a shape of 3 dimensions!'
        assert radius > 0 and np.shape(radius) == (), 'Radius has to be positive scalar value!'
        mag_shape = np.array([[[np.sqrt(  (x-center[2])**2 
                                        + (y-center[1])**2 
                                        + (z-center[0])**2 ) <= radius
                            for x in range(dim[2])] 
                            for y in range(dim[1])] 
                            for z in range(dim[0])])
        return mag_shape
    
    @classmethod
    def filament(cls, dim, pos, axis='y'):
        '''Get the magnetic shape of a single filament in x-, y-, or z-direction.
        Arguments:
            dim  - the dimensions of the grid, shape(z, y, x)
            pos  - the position of the filament in pixel coordinates, shape(coord1, coord2)
            axis - the orientation of the filament axis, (String: 'x', 'y', 'z'), default = 'y'
        Returns:
            the magnetic shape as a 3D-array with ones and zeros
            
        '''
        assert np.shape(dim) == (3,), 'Parameter dim has to be a shape of 3 dimensions!'
        assert np.shape(pos) == (2,), 'Parameter pos has to be a shape of 2 dimensions!'
        assert axis == 'z' or axis == 'y' or axis == 'x', 'Axis has to be x, y or z (as String)!'
        mag_shape = np.zeros(dim)
        if axis == 'z':        
            mag_shape[:, pos[0], pos[1]] = 1
        elif axis == 'y':
            mag_shape[pos[0], :, pos[1]] = 1
        elif axis == 'x':
            mag_shape[pos[0], pos[1], :] = 1
        return mag_shape
    
    @classmethod    
    def pixel(cls, dim, pixel):
        '''Get the magnetic shape of a single magnetized pixel.
        Arguments:
            dim   - the dimensions of the grid, shape(z, y, x)
            pixel - the coordinates of the magnetized pixel, shape(z, y, x)
        Returns:
            the magnetic shape as a 3D-array with ones and zeros
            
        '''
        assert np.shape(dim)   == (3,), 'Parameter dim has to be a shape of 3 dimensions!'
        assert np.shape(pixel) == (3,), 'Parameter pixel has to be a shape of 3 dimensions!'
        mag_shape = np.zeros(dim)
        mag_shape[pixel] = 1
        return mag_shape
    

def create_mag_dist(mag_shape, beta, magnitude=1):
    '''Create a 3-dimensional magnetic distribution of a homogeneous magnetized object.
    Arguments:
        mag_shape - the magnetic shapes (numpy arrays, see Shapes.* for examples)
        beta      - the angle, describing the direction of the magnetized object
        magnitude - the relative magnitudes for the magnetic shape (optional, one if not specified)
    Returns:
        the 3D magnetic distribution as a MagData object (see pyramid.magdata for reference)
        
    '''
    dim = np.shape(mag_shape)
    assert len(dim) == 3, 'Magnetic shapes must describe 3-dimensional distributions!'
    z_mag = np.zeros(dim)  # TODO: Implement another angle!
    y_mag = np.ones(dim) * np.sin(beta) * mag_shape * magnitude
    x_mag = np.ones(dim) * np.cos(beta) * mag_shape * magnitude
    return z_mag, y_mag, x_mag


def create_mag_dist_comb(mag_shape_list, beta_list, magnitude_list=None):
    '''Create a 3-dimensional magnetic distribution from a list of homogeneous magnetized objects.
    Arguments:
        mag_shape_list - a list of magnetic shapes (numpy arrays, see Shapes.* for examples)
        beta_list      - a list of angles, describing the direction of the magnetized objects
        magnitude_list - a list of relative magnitudes for the magnetic shapes
                         (optional, if not specified, every relative magnitude is set to one)
    Returns:
        the 3D magnetic distribution as a MagData object (see pyramid.magdata for reference)
        
    '''
    # If no relative magnitude is specified, 1 is assumed for every homog. object:
    if magnitude_list is None:
        magnitude_list = np.ones(len(beta_list))    
    # For every shape of a homogeneous object a relative magnitude and angle have to be set:
    assert np.shape(mag_shape_list)[0] == len(beta_list) == len(magnitude_list), \
           'Lists have not the same length!'
    dim = np.shape(mag_shape_list[0])  # Has to be the shape of ALL mag_shapes!
    assert len(dim) == 3, 'Magnetic shapes must describe 3-dimensional distributions!'
    assert np.array([mag_shape_list[i].shape == dim for i in range(len(mag_shape_list))]).all(), \
           'Magnetic shapes must describe distributions with the same size!'
    # Start with a zero distribution:
    x_mag = np.zeros(dim)
    y_mag = np.zeros(dim)
    z_mag = np.zeros(dim)
    # Add every specified homogeneous object:
    for i in range(np.size(beta_list)):
        mag_object = create_mag_dist(mag_shape_list[i], beta_list[i], magnitude_list[i])    
        z_mag += mag_object[0]
        y_mag += mag_object[1]
        x_mag += mag_object[2]
    return z_mag, y_mag, x_mag