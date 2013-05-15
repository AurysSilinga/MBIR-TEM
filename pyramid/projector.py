# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:44:36 2013

@author: Jan
"""# TODO: Docstring

from magdata import MagData

def simple_axis_projection(mag_data, axis=0):
    # TODO: assert isinstance(mag_data, MagData), 'Object is no instance of MagData!'
    return (mag_data.magnitude[0].mean(axis),  # x_mag
            mag_data.magnitude[1].mean(axis),  # y_mag
            mag_data.magnitude[2].mean(axis))  # z_mag
    
    
# TODO: proper projection algorithm with two angles and such!