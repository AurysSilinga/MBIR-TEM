# -*- coding: utf-8 -*-
"""Create PhaseMaps from MagDatas via simple axis projection."""


import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
filename = 'magdata_mc_array_sphere_disc_slab.nc'
b_0 = 1.
dim_uv = None
###################################################################################################

mag_data = py.VectorData.load_from_netcdf4(filename)
phase_map = py.pm(mag_data, dim_uv=dim_uv, b_0=b_0)
phase_map.save_to_netcdf4('phasemap_{}'.format(filename.replace('magdata_', '')))
phase_map.display_combined()
