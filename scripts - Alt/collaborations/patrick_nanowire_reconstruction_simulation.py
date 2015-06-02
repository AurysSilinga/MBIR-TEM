# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 14:17:56 2014

@author: Jan
"""

from PIL import Image
import numpy as np

from jutil.taketime import TakeTime
from pyramid import *  # analysis:ignore

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)

###################################################################################################
gain = 5
b_0 = 1
lam = 1E-3
length = 300
cutoff = 50
trans = 40
pads = [100, 250, 500, 800]
INPATH = '../../output/vtk data/longtube_withcap/'
OUTPATH = '../../output/patrick/'
FILE = 'CoFeB_tube_cap_4nm_lying_down'
###################################################################################################

# Read in big 3D files:
mag_data_3d = MagData.load_from_netcdf4(INPATH+FILE+'.nc')
a = mag_data_3d.a
dim_3d = mag_data_3d.dim

# Make 2D projection:
dim_uv_big = (dim_3d[1]+200, dim_3d[2]+200)
projector = SimpleProjector(dim_3d, dim_uv=dim_uv_big)
mag_data_big = projector(mag_data_3d)
phase_map_big = pm(mag_data_big, dim_uv=dim_uv_big)
phase_map_big.display_combined(gain=gain)
mask_big = mag_data_big.get_mask()
dim_big = mag_data_big.dim
dim_uv_big = phase_map_big.dim_uv

# Reconstruction of complete wire:
data_set = DataSet(a, dim_big, b_0, mask_big)
data_set.append(phase_map_big, SimpleProjector(dim_big))
regularisator = FirstOrderRegularisator(mask_big, lam, p=2)
with TakeTime('reconstruction time'):
    mag_data_big_rec = rc.optimize_linear(data_set, regularisator=regularisator, max_iter=50)[0]
phase_map_big_rec = pm(mag_data_big_rec)
axis_p, axis_h = phase_map_big.display_combined('Reconstruction (complete wire)', gain=gain)
axis_p.axhline(y=length, linewidth=1, color='b', linestyle='-')
axis_h.axhline(y=length, linewidth=1, color='b', linestyle='-')
plt.savefig(OUTPATH+FILE+'RECON_COMPLETE.png'.format(pad), dpi=500)
plt.close('all')

for pad in pads:
    # Make smaller cutout with optional padding:
    mask = mask_big[:, :length+pad, :].copy()
    mask[:, mask.shape[1]-cutoff:, :] = False
    dim = mask.shape
    dim_uv = (dim[1], dim[2])
    phase = np.zeros(dim_uv)
    phase[:length, :] = phase_map_big.phase[:length, :]
    phase_map = PhaseMap(a, phase)
    phase_map_orig = PhaseMap(a, phase[:length, :])

    # Create DataSet and regularisator:
    data_set = DataSet(a, dim, b_0, mask)
    data_set.append(phase_map, SimpleProjector(dim))

    # Create Se_inv
    mask_Se = np.ones(dim_uv)
    mask_Se[length:length+pad, :] = 0
    ramp = np.tile(np.linspace(1, 0, trans), mask_Se.shape[1]).reshape((mask_Se.shape[1], trans)).T
    mask_Se[length-trans:length, :] = ramp
    data_set.set_Se_inv_diag_with_masks([mask_Se])

    # Create regularisator:
    regularisator = FirstOrderRegularisator(mask, lam, p=2)

    # Reconstruct:
    with TakeTime('reconstruction time'):
        mag_data_rec = rc.optimize_linear(data_set, regularisator=regularisator, max_iter=50)[0]

    # Calculate additional PhaseMaps:
    phase_map_rec = pm(mag_data_rec)
    phase_map_cut = PhaseMap(a, phase_map_rec.phase[:length, :])
    phase_map_diff = phase_map_orig - phase_map_cut

    # Plotting:
    phase_map_orig.display_combined('Original distribution', gain=gain)
    plt.savefig(OUTPATH+FILE+'_{}pxpadding_ORIG.png'.format(pad), dpi=500)
    axis_p, axis_h = phase_map.display_combined('Original Distribution (padded)', gain=gain)
    axis_p.axhline(y=length, linewidth=1, color='b', linestyle='-')
    axis_h.axhline(y=length, linewidth=1, color='b', linestyle='-')
    axis_p.axhline(y=length-trans, linewidth=1, color='r', linestyle='--')
    axis_h.axhline(y=length-trans, linewidth=1, color='r', linestyle='--')
    plt.savefig(OUTPATH+FILE+'_{}ramp_{}pxpadding_ORIG_PAD.png'.format(trans, pad), dpi=500)
    axis_p, axis_h = phase_map_rec.display_combined('Reconstr. Distribution (padded)', gain=gain)
    axis_p.axhline(y=length, linewidth=1, color='b', linestyle='-')
    axis_h.axhline(y=length, linewidth=1, color='b', linestyle='-')
    axis_p.axhline(y=length-trans, linewidth=1, color='r', linestyle='--')
    axis_h.axhline(y=length-trans, linewidth=1, color='r', linestyle='--')
    plt.savefig(OUTPATH+FILE+'_{}ramp_{}pxpadding_RECON_PAD.png'.format(trans, pad), dpi=500)
    phase_map_cut.display_combined('Reconstr. Distribution', gain=gain)
    plt.savefig(OUTPATH+FILE+'_{}ramp_{}pxpadding_RECON.png'.format(trans, pad))
    phase_map_diff.display_combined('Difference')
    plt.savefig(OUTPATH+FILE+'_{}ramp_{}pxpadding_DIFF.png'.format(trans, pad))
    axis = mag_data_rec.quiver_plot(ar_dens=4, log=True)
    axis.axhline(y=length, linewidth=1, color='b', linestyle='-')
    axis.axhline(y=length-trans, linewidth=1, color='r', linestyle='--')
    plt.savefig(OUTPATH+FILE+'_{}ramp_{}pxpadding_MAG.png'.format(trans, pad), dpi=500)
    plt.close('all')
