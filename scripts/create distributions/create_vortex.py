#! python
# -*- coding: utf-8 -*-
"""Create the Pyramid-Logo."""


import pdb
import traceback
import sys
import os

import matplotlib.pyplot as plt

import pyramid.magcreator as mc
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PMAdapterFM
#import pyramid.holoimage as hi
from pyramid.magdata import MagData


def create_vortex():
    '''Calculate and display the Pyramid-Logo.
    Arguments:
        None
    Returns:
        None

    '''
    directory = '../../output/magnetic distributions'
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Input parameters:
    filename = directory + '/mag_dist_vortex.txt'
    a = 10.0  # in nm
#    density = 1
    dim = (64, 64, 64)
    center = (int(dim[0]/2)-0.5, int(dim[1]/2)-0.5, int(dim[2]/2)-0.5)
    radius = 0.25 * dim[1]
    height = dim[0]/4
    # Create magnetic shape:
    mag_shape = mc.Shapes.disc(dim, center, radius, height)
    mag_data = MagData(a, mc.create_mag_dist_vortex(mag_shape))
    mag_data.quiver_plot()
    mag_data.save_to_llg(filename)
    projector = SimpleProjector(dim)
    phase_map = PMAdapterFM(a, projector)(mag_data)
    phase_map.display_combined()
#    hi.display_combined(phase_map, density, 'Vortex State')
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
