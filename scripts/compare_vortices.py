#! python
# -*- coding: utf-8 -*-
"""Create the Pyramid-Logo."""


import pdb
import traceback
import sys
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

import pyramid.magcreator as mc
import pyramid.projector as pj
import pyramid.phasemapper as pm
import pyramid.holoimage as hi
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap


PHI_0 = -2067.83  # magnetic flux in T*nm²


def compare_vortices():
    '''Calculate and display the Pyramid-Logo.
    Arguments:
        None
    Returns:
        None

    '''
    # Input parameters:
    dim_list = [(16, 256, 256), (8, 128, 128), (4, 64, 64), (2, 32, 32), (1, 16, 16)]
    res_list = [1., 2., 4., 8., 16.]  # in nm
    density = 20

    x = []
    y = []

    # Starting magnetic distribution:
    dim_start = (2*dim_list[0][0], 2*dim_list[0][1], 2*dim_list[0][2])
    res_start = res_list[0] / 2
    center = (dim_start[0]/2 - 0.5, dim_start[1]/2 - 0.5, dim_start[2]/2 - 0.5)
    radius = 0.25 * dim_start[1]
    height = 0.5 * dim_start[0]
    mag_shape = mc.Shapes.disc(dim_start, center, radius, height)
    mag_data = MagData(res_start, mc.create_mag_dist_vortex(mag_shape))

    # Analytic solution:
    L = dim_list[0][1]  # in px/nm
    Lz = 0.5 * dim_list[0][0]  # in px/nm
    R = 0.25 * L  # in px/nm
    x0 = L / 2  # in px/nm

    def F(x):
        coeff = pi*Lz/PHI_0
        result = coeff * np.where(np.abs(x - x0) <= R, (np.abs(x-x0)-R), 0)
        return result
    x_an = np.linspace(0, L, 1001)
    y_an = F(x_an)

    for i, (dim, res) in enumerate(zip(dim_list, res_list)):
        # Create coarser grid for the magnetization:
        print 'dim = ', dim, 'res = ', res
        z_mag = mag_data.magnitude[0].reshape(dim[0], 2, dim[1], 2, dim[2], 2)
        y_mag = mag_data.magnitude[1].reshape(dim[0], 2, dim[1], 2, dim[2], 2)
        x_mag = mag_data.magnitude[2].reshape(dim[0], 2, dim[1], 2, dim[2], 2)
        magnitude = (z_mag.mean(axis=5).mean(axis=3).mean(axis=1),
                     y_mag.mean(axis=5).mean(axis=3).mean(axis=1),
                     x_mag.mean(axis=5).mean(axis=3).mean(axis=1))
        mag_data = MagData(res, magnitude)
        #mag_data.quiver_plot()
        projection = pj.simple_axis_projection(mag_data)
        phase_map = PhaseMap(res, pm.phase_mag_real(res, projection, 'slab'))
        hi.display_combined(phase_map, density, 'Vortex State, res = {}'.format(res))
        x.append(np.linspace(0, dim[1]*res, dim[1]))
        y.append(phase_map.phase[dim[1]/2, :])

    # Plot:
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(x[0], y[0], 'r',
              x[1], y[1], 'm',
              x[2], y[2], 'y',
              x[3], y[3], 'g',
              x[4], y[4], 'c',
              x_an, y_an, 'k')
    axis.set_xlabel('x [nm]')
    axis.set_ylabel('phase')


if __name__ == "__main__":
    try:
        compare_vortices()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
