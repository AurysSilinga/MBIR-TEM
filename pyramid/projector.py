# -*- coding: utf-8 -*-
"""Planar projection of the magnetization distribution of a MagData object."""


from pyramid.magdata import MagData


def simple_axis_projection(mag_data, axis='z'):
    '''Project a magnetization distribution along one of the main axes of the 3D-grid.
    Arguments:
        mag_data - a MagData object storing the magnetization distribution
        axis     - the projection direction (String: 'x', 'y', 'z'), default = 'z'
    Returns:
        the in-plane projection of the magnetization as a tuple: (x_mag, y_mag)
        ()
    
    '''
    assert isinstance(mag_data, MagData), 'Parameter mag_data has to be a MagData object!'
    assert axis == 'z' or axis == 'y' or axis == 'x', 'Axis has to be x, y or z (as String)!'
    if axis == 'z':
        projection = (mag_data.magnitude[1].mean(0) * mag_data.dim[0],  # y_mag -> v_mag
                      mag_data.magnitude[2].mean(0) * mag_data.dim[0])  # x_mag -> u_mag
    elif axis == 'y':
        projection = (mag_data.magnitude[0].mean(1) * mag_data.dim[1],  # z_mag -> v_mag
                      mag_data.magnitude[2].mean(1) * mag_data.dim[1])  # x_mag -> u_mag
    elif axis == 'x':
        projection = (mag_data.magnitude[0].mean(2) * mag_data.dim[2],  # y_mag -> v_mag
                      mag_data.magnitude[1].mean(2) * mag_data.dim[2])  # x_mag -> u_mag
    return projection
    
    
# TODO: proper projection algorithm with two angles and such!
# CAUTION: the res for the projection does not have to be the res of the 3D-magnetization!
# Just for a first approach