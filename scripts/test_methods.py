#! python
# -*- coding: utf-8 -*-
"""Compare the different methods to create phase maps."""

import pdb
import traceback
import sys

from numpy import pi

import pyramid.magcreator as mc
import pyramid.projector as pj
import pyramid.phasemapper as pm
import pyramid.holoimage as hi
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap


def compare_methods():
    '''Calculate and display the phase map from a given magnetization.
    Arguments:
        None
    Returns:
        None

    '''
    # Input parameters:
    res = 10.0  # in nm
    phi = 0
    theta = pi/2
    tilt = pi/4
    density = 0.25
    dim = (128, 128, 128)  # in px (z, y, x)
    # Create magnetic shape:
    geometry = 'sphere'
    if geometry == 'slab':
        center = (dim[0]/2-0.5, dim[1]/2-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts at 0!
        width = (dim[0]/3., dim[1]/4., dim[2]/2.)  # in px (z, y, x)
        mag_shape = mc.Shapes.slab(dim, center, width)
    elif geometry == 'disc':
        center = (dim[0]/2-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts at 0!
        radius = dim[1]/4  # in px
        height = dim[0]/2  # in px
        mag_shape = mc.Shapes.disc(dim, center, radius, height)
    elif geometry == 'sphere':
        center = (dim[0]/2, dim[1]/2, dim[2]/2)  # in px (z, y, x) index starts with 0!
        radius = dim[0]/4  # in px
        mag_shape = mc.Shapes.sphere(dim, center, radius)
    # Project the magnetization data:
    mag_data = MagData(res, mc.create_mag_dist_homog(mag_shape, phi, theta))

#    density = 0.3
#    mag_data = MagData.load_from_llg('../output/magnetic distributions/mag_dist_sphere.txt')

    res = mag_data.res
    mag_data.quiver_plot(ax_slice=dim[1]/2)
#    mag_data.quiver_plot3d()
    import time
    start = time.time()
    projection = pj.single_tilt_projection(mag_data, tilt)
    print 'Total projection time:', time.time() - start
    # Construct phase maps:
    phase_map_mag = PhaseMap(res, pm.phase_mag_fourier(res, projection, padding=1))
    phase_map_elec = PhaseMap(res, pm.phase_elec(res, projection, v_0=3))
    # Display the combinated plots with phasemap and holography image:
    hi.display_combined(phase_map_mag, density, title='Magnetic Phase')
    hi.display_combined(phase_map_elec, density, title='Electric Phase')

    phase_map = PhaseMap(res, phase_map_mag.phase+phase_map_elec.phase)
    hi.display_combined(phase_map, density)
    
    import matplotlib.pyplot as plt
    x = range(dim[2])
    y = phase_map_elec.phase[dim[1]/2, :]
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(x, y, label='Real space method')
    axis.grid()
    

if __name__ == "__main__":
    try:
        compare_methods()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
