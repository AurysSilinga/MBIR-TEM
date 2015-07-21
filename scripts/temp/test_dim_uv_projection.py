# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 19:11:33 2015

@author: Jan
"""

import numpy as np
import pyramid as pr


dim = (1, 5, 5)
dim_uv = None
width = (1, 2, 2)
center = (0.5, 2.5, 2.5)
mag_shape = pr.magcreator.Shapes.slab(dim, center, width)
mag_data = pr.MagData(10, pr.magcreator.create_mag_dist_homog(mag_shape, np.pi/4))
mag_data.quiver_plot()
projector = pr.SimpleProjector(dim, axis='z', dim_uv=dim_uv)
mag_proj = projector(mag_data)
mag_proj.quiver_plot()
pr.pm(mag_proj).display_combined()
weight = np.array(projector.weight.todense())
pr.an.phase_mag_vortex(dim, 10, center, 12.5, 1).display_combined()
