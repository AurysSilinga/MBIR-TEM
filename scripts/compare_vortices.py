# -*- coding: utf-8 -*-
"""Create the Pyramid-Logo."""


import pdb, traceback, sys
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

import pyramid.magcreator  as mc
import pyramid.projector   as pj
import pyramid.phasemapper as pm
import pyramid.holoimage   as hi
from pyramid.magdata  import MagData
from pyramid.phasemap import PhaseMap


def compare_vortices():
    '''Calculate and display the Pyramid-Logo.
    Arguments:
        None
    Returns:
        None
    
    '''
    # Input parameters:  
    dim_list = [(1, 512, 512), (1, 256, 256), (1, 128, 128), (1, 64, 64), (1, 32, 32), (1, 16, 16)]
    res_list = [2., 4., 8., 16., 32., 64.]  # in nm
    density = 10
    height = 1
    
    x = []
    y = []
    
    # Starting magnetic distribution:
    dim_start = (1, 2*dim_list[0][1], 2*dim_list[0][2])
    res_start = res_list[0] / 2
    print 'dim_start = ', dim_start
    print 'res_start = ', res_start
    center = (0, dim_start[1]/2 - 0.5, dim_start[2]/2 - 0.5)
    radius = 0.25 * dim_start[1]
    mag_shape = mc.Shapes.disc(dim_start, center, radius, height)
    mag_data = MagData(res_start, mc.create_mag_dist_vortex(mag_shape))
    #mag_data.quiver_plot()
    
    for i, (dim, res) in enumerate(zip(dim_list, res_list)):
        # Create coarser grid for the magnetization:
        z_mag = mag_data.magnitude[0].reshape(1, dim[1], 2, dim[1], 2)
        y_mag = mag_data.magnitude[1].reshape(1, dim[1], 2, dim[1], 2)
        x_mag = mag_data.magnitude[2].reshape(1, dim[1], 2, dim[1], 2)
        print 'dim = ', dim
        print 'res = ', res
        magnitude = (z_mag.mean(axis=4).mean(axis=2) / 2,
                     y_mag.mean(axis=4).mean(axis=2) / 2,
                     x_mag.mean(axis=4).mean(axis=2) / 2)
        mag_data = MagData(res, magnitude)
        print np.shape(mag_data.magnitude[0])
        #mag_data.quiver_plot()
        projection = pj.simple_axis_projection(mag_data)
        phase_map = PhaseMap(res, pm.phase_mag_real(res, projection, 'slab'))    
        hi.display_combined(phase_map, density, 'Vortex State, res = {}'.format(res))
        x.append(np.linspace(0, dim[1]*res, dim[1]))
        y.append(phase_map.phase[dim[1]/2, :]/pi)
        fig = plt.figure()
        fig.add_subplot(1, 1, 1)
        plt.plot(x[i], y[i])
#    
    # Plot:
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(x[0], y[0], 'k',
              x[1], y[1], 'r',
              x[2], y[2], 'm',
              x[3], y[3], 'y',
              x[4], y[4], 'c',
              x[5], y[5], 'g',
              x[6], y[6], 'b')
    axis.set_xlabel('x [nm]')
    axis.set_ylabel('phase')
    
if __name__ == "__main__":
    try:
        compare_vortices()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)