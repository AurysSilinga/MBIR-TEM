# -*- coding: utf-8 -*-
"""Create PhaseMaps from MagDatas via simple axis projection."""


import numpy as np
import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
filename = 'magdata_mc_paper2_homog_slab_simulated_dim=(1, 512, 512).nc'
b_0 = 1.
dim_uv = None
rotation = np.pi/4.
tilt = 0.
###################################################################################################

mag_data = py.VectorData.load_from_netcdf4(filename)
projector = py.RotTiltProjector(mag_data.dim, rotation, tilt, dim_uv)
mag_proj = projector(mag_data)
phasemapper = py.PhaseMapperRDFC(py.Kernel(mag_data.a, projector.dim_uv, b_0))
phase_map = phasemapper(mag_proj)
phase_map.mask = mag_proj.get_mask()[0, ...]
phase_map.save_to_netcdf4('phasemap_{}'.format(filename.replace('magdata_', '')))
phase_map.display_combined()
