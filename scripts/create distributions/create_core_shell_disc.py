# -*- coding: utf-8 -*-
"""Create a core-shell disc."""


import pdb
import traceback
import sys
import os

import numpy as np

import matplotlib.pyplot as plt

import pyramid.magcreator as mc
import pyramid.projector as pj
import pyramid.phasemapper as pm
import pyramid.holoimage as hi
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap


def create_core_shell_disc():
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
    filename = directory + '/mag_dist_core_shell_disc.txt'
    res = 1.0  # in nm
    density = 1
    dim = (32, 32, 32)
    center = (dim[0]/2-0.5, int(dim[1]/2)-0.5, int(dim[2]/2)-0.5)
    radius_core = dim[1]/8
    radius_shell = dim[1]/4
    height = dim[0]/2
    # Create magnetic shape:
    mag_shape_core = mc.Shapes.disc(dim, center, radius_core, height)
    mag_shape_outer = mc.Shapes.disc(dim, center, radius_shell, height)
    mag_shape_shell = np.logical_xor(mag_shape_outer, mag_shape_core)
    mag_data = MagData(res, mc.create_mag_dist_vortex(mag_shape_shell, magnitude=0.75))
    mag_data.add_magnitude(mc.create_mag_dist_homog(mag_shape_core, phi=0, theta=0))
    mag_data.quiver_plot('z-projection', proj_axis='z')
    mag_data.quiver_plot('x-projection', proj_axis='x')
    mag_data.quiver_plot3d()
    mag_data.save_to_llg(filename)
    projection_z = pj.simple_axis_projection(mag_data, axis='z')
    projection_x = pj.simple_axis_projection(mag_data, axis='x')
    phase_map_z = PhaseMap(res, pm.phase_mag_real(res, projection_z, 'slab'))
    phase_map_x = PhaseMap(res, pm.phase_mag_real(res, projection_x, 'slab'))
    hi.display_combined(phase_map_z, density, 'Core-Shell structure (z-projection)')
    phase_axis, holo_axis = hi.display_combined(phase_map_x, density, 
                                                'Core-Shell structure (x-projection)')
    phase_axis.set_xlabel('y [nm]')
    phase_axis.set_ylabel('z [nm]')
    holo_axis.set_xlabel('y-axis [px]')
    holo_axis.set_ylabel('z-axis [px]')
    phase_slice_z = phase_map_z.phase[dim[1]/2, :]
    phase_slice_x = phase_map_x.phase[dim[0]/2, :]
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.plot(range(dim[2]), phase_slice_z)
    plt.title('Phase slice along x for z-projection')
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.plot(range(dim[1]), phase_slice_x)
    plt.title('Phase slice along y for x-projection')


if __name__ == "__main__":
    try:
        create_core_shell_disc()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
