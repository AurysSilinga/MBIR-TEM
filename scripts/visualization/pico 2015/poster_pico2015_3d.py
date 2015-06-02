# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:27:39 2015

@author: Jan
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
b_0 = 0.6
lam = 1E-4
ar_dens = 10
###################################################################################################

directory = os.path.join(py.DIR_FILES, 'images', 'poster PICO 2015')
if not os.path.exists(directory):
    os.makedirs(directory)


# 3D Stuff: #######################################################################################

# Load magnetization distribution:
mag_data = py.MagData.load_from_netcdf4('magdata_mc_array_sphere_disc_slab.nc')
mag_data_rec = py.MagData.load_from_netcdf4('magdata_rec_mc_array_sphere_disc_slab.nc')
dim = mag_data.dim

# Construct data set and regularisator:
data = py.DataSet(mag_data_rec.a, mag_data_rec.dim, b_0)

# Construct projectors:
projectors = []
angles = np.linspace(-90, 90, num=7)
for angle in angles:
    angle_rad = angle*np.pi/180
    projectors.append(py.XTiltProjector(mag_data.dim, angle_rad))
    projectors.append(py.YTiltProjector(mag_data.dim, angle_rad))

# Add projectors and construct according phase maps:
data.projectors = projectors
data.phase_maps = data.create_phase_maps(mag_data)

# Construct mask:
data.set_3d_mask()  # Construct mask from 2D phase masks!

# Plot stuff:
data.display_mask(ar_dens=np.ceil(np.max(dim)/32.))
#plt.savefig(os.path.join(directory, 'constructed mask.png'))#,
##            bbox_inches='tight', dpi=300)
mag_data.quiver_plot3d('Original Distribution', ar_dens=np.ceil(np.max(dim)/16.), limit=1.2)
figure = mag_data_rec.quiver_plot3d('Reconstructed Distribution', ar_dens=np.ceil(np.max(dim)/16.),
                                    limit=1.2)

plt.close('all')

data.display_phase()
for n in plt.get_fignums():
    plt.figure(n).savefig(os.path.join(directory, 'phase map {}'.format(n)),
                          bbox_inches='tight', dpi=300)

#plt.close('all')
