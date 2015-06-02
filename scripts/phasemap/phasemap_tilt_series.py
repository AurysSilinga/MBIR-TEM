# -*- coding: utf-8 -*-
"""Create PhaseMaps from MagDatas via simple axis projection."""


import numpy as np
import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
filename = 'magdata_mc_homog_sphere_dim=(32, 32, 32).nc'
b_0 = 1.
axis = 'z'
dim_uv = None

angles = np.linspace()
###################################################################################################

mag_data = py.MagData.load_from_netcdf4(filename)
phase_map = py.pm(mag_data, axis=axis, dim_uv=dim_uv, b_0=b_0)
phase_map.save_to_netcdf4('phasemap_{}_axis={}'.format(filename.replace('magdata_', ''), axis))
phase_map.display_combined()








dim = mag_data.dim
dim_uv = (500, 200)
angles = np.arange(-60, 61, 5)#[0, 20, 40, 60]

#mag_data_xy = mag_data.copy()
#mag_data_xy.magnitude[2] = 0
#
#mag_data_z = mag_data.copy()
#mag_data_z.magnitude[0] = 0
#mag_data_z.magnitude[1] = 0

# Iterate over all angles:
for angle in angles:
    angle_rad = np.pi/2 + angle*np.pi/180
    projector = XTiltProjector(dim, angle_rad, dim_uv)
    mag_proj = projector(mag_data)
    phase_map = PhaseMapperRDFC(Kernel(mag_data.a, projector.dim_uv))(mag_proj)
    phase_map.display_phase('Phase Map Nanowire Tip', cmap='gray')
    plt.savefig(PATH+'_nanowire_xtilt_{}.png'.format(angle), dpi=500)
#    phase_map = PhaseMapperRDFC(Kernel(mag_data.a, projector.dim_uv))(mag_proj)
#    phase_map.display_combined('Phase Map Nanowire Tip', gain=gain,
#                               interpolation='bilinear')
#    plt.savefig(PATH+'_nanowire_xtilt_{}.png'.format(angle), dpi=500)
#    mag_proj.scale_down(2)
#    axis = mag_proj.quiver_plot()
#    plt.savefig(PATH+'_nanowire_mag_xtilt_{}.png'.format(angle), dpi=500)
#    axis = mag_proj.quiver_plot(log=True)
#    plt.savefig(PATH+'_nanowire_mag_log_xtilt_{}.png'.format(angle), dpi=500)
    # Close plots:
    plt.close('all')
    gc.collect()
    print 'RSS = {:.2f} MB'.format(proc.memory_info().rss/1024.**2)
    print angle, 'deg. done!'