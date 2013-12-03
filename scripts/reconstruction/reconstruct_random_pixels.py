# -*- coding: utf-8 -*-
"""Create random magnetic distributions."""


import random as rnd
from numpy import pi

import pyramid.magcreator as mc
from pyramid.magdata import MagData
import pyramid.phasemapper as pm
import pyramid.projector as pj
import pyramid.holoimage as hi
from pyramid.phasemap import PhaseMap
import pyramid.reconstructor as rc


# Input parameters:
n_pixel = 5
dim = (1, 16, 16)
b_0 = 1  # in T
a = 10.0  # in nm
rnd.seed(18)

# Create empty MagData object and add random pixels:
mag_data = MagData(a)
for i in range(n_pixel):
    pixel = (rnd.randrange(dim[0]), rnd.randrange(dim[1]), rnd.randrange(dim[2]))
    mag_shape = mc.Shapes.pixel(dim, pixel)
    phi = 2 * pi * rnd.random()
    magnitude = rnd.random()
    mag_data.add_magnitude(mc.create_mag_dist_homog(mag_shape, phi, magnitude))
# Plot magnetic distribution, phase map and holographic contour map:
mag_data.quiver_plot()
projection = pj.simple_axis_projection(mag_data)
phase_map = PhaseMap(a, pm.phase_mag(a, projection, b_0))
hi.display_combined(phase_map, 10, 'Generated Distribution')

# Reconstruct the magnetic distribution:
mag_data_rec = rc.reconstruct_simple_leastsq(phase_map, mag_data.get_mask(), b_0)

# Display the reconstructed phase map and holography image:
projection_rec = pj.simple_axis_projection(mag_data_rec)
phase_map_rec = PhaseMap(a, pm.phase_mag(a, projection_rec, b_0))
hi.display_combined(phase_map_rec, 10, 'Reconstructed Distribution')
