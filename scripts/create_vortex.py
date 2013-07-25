# -*- coding: utf-8 -*-
"""Create the Pyramid-Logo."""


import pdb
import traceback
import sys
import matplotlib.pyplot as plt

import pyramid.magcreator as mc
import pyramid.projector as pj
import pyramid.phasemapper as pm
import pyramid.holoimage as hi
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap


def create_vortex():
    '''Calculate and display the Pyramid-Logo.
    Arguments:
        None
    Returns:
        None

    '''
    # Input parameters:
    filename = '../output/mag_dist_vortex.txt'
    res = 10.0  # in nm
    density = 1
    dim = (1, 128, 128)
    center = (0, int(dim[1]/2)-0.5, int(dim[2]/2)-0.5)
    radius = 0.25 * dim[1]
    height = 1
    # Create magnetic shape:
    mag_shape = mc.Shapes.disc(dim, center, radius, height)
    mag_data = MagData(res, mc.create_mag_dist_vortex(mag_shape))
    mag_data.quiver_plot()
    mag_data.quiver_plot3d()
    mag_data.save_to_llg(filename)
    projection = pj.simple_axis_projection(mag_data)
    phase_map = PhaseMap(res, pm.phase_mag_real(res, projection, 'slab'))
    hi.display_combined(phase_map, density, 'Vortex State')
    phase_slice = phase_map.phase[dim[1]/2, :]
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(range(dim[1]), phase_slice)


if __name__ == "__main__":
    try:
        create_vortex()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
