# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:31:59 2015

@author: Jan
"""


import os
import numpy as np
import hyperspy.hspy as hp
import pyramid as pr


phase_map = pr.PhaseMap.load_from_netcdf4('phasemap_tif_martial_magnetite.nc')
mag_rec = pr.MagData.load_from_netcdf4('magdata_rec_tif_martial_magnetite_lam=0.001.nc')
ramp_params = (0.350598105531, 0.00022362574109514882, 0.00075016020407003121)
phase_ramp = pr.Ramp.create_ramp(phase_map.a, phase_map.dim_uv, ramp_params)
phase_corr = phase_map - phase_ramp
phase_rec = pr.pm(mag_rec)
phase_diff = phase_rec - phase_corr

#mag_rec.quiver_plot('Reconstructed distribution', ar_dens=4)
#phase_map.display_combined('Original Input')
phase_corr.display_combined('Input Corrected')
#phase_ramp.display_phase('Ramp')
#phase_rec.display_combined('Reconstructed Phase')
#phase_diff.display_phase('Difference')

#mag_rec.save_to_llg(os.getcwd()+'/mag_rec.txt')
#np.save('phase_input_corr.npy', phase_corr.phase)
#np.savetxt('phase_input_corr.txt', phase_corr.phase)

phase_corr_hp = hp.signals.Image(phase_map.phase)

phase_corr_hp.axes_manager[0].name = 'X'
phase_corr_hp.axes_manager[0].units = 'nm'
phase_corr_hp.axes_manager[0].scale = phase_corr.a
phase_corr_hp.axes_manager[1].name = 'Y'
phase_corr_hp.axes_manager[1].units = 'nm'
phase_corr_hp.axes_manager[1].scale = phase_corr.a
#phase_corr_hp.save('phase_input_corr.rpl')
phase_corr_hp.plot(axes_ticks=True)
