# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:45:24 2015

@author: Jan
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import pyramid as py


###################################################################################################
filename = 'magdata_vtk_CoFeB_tube_cap_4nm.nc'
###################################################################################################

mag_data = py.MagData.load_from_netcdf4(filename)
mag_data.quiver_plot3d(ar_dens=8)
mag_x, mag_y, mag_z = mag_data.magnitude
mag_r = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)

cos_angles = mag_z/mag_r
cos_angles_mean = np.nanmean(cos_angles, axis=(1, 2))

plt.plot(mag_data.a*np.arange(mag_data.dim[0]), cos_angles_mean)

data = np.vstack([mag_data.a*np.arange(mag_data.dim[0]), cos_angles_mean]).T
np.savetxt(os.path.join(py.DIR_FILES, filename.replace('nc', 'txt')), data)
