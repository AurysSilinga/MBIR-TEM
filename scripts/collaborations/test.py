# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 08:49:48 2014

@author: Jan
"""


from numpy import pi
import numpy as np

import pyramid.magcreator as mc
from pyramid.magdata import MagData


dim = (1, 32, 32)
center = (0, 16, 16)
width = (1, 8, 16)
mag_data = MagData(1, mc.create_mag_dist_homog(mc.Shapes.slab(dim, center, width), pi/6))
mag_data.quiver_plot()
mag_data.quiver_plot3d()
x, y, z = mag_data.magnitude[:, 0, ...]
x_rot = np.rot90(mag_data.magnitude[0, 0, ...]).copy()
y_rot = np.rot90(mag_data.magnitude[1, 0, ...]).copy()
#z_rot = np.rot90(mag_data.magnitude[2, 0, ...]).copy()
mag_data.magnitude[0, 0, ...] = -y_rot
mag_data.magnitude[1, 0, ...] = x_rot
mag_data.quiver_plot()
mag_data.quiver_plot3d()
