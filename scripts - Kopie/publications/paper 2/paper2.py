# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 10:57:53 2015

@author: Jan
"""


import numpy as np

from pyramid import *  # analysis:ignore
from pyramid import magcreator as mc
from pyramid import reconstruction as rc

from jutil.taketime import TakeTime

from matplotlib.patches import Rectangle

from PIL import Image

import pickle

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)


def calc_diagnostics(pos, mag_data_opt, cost, max_iter=2000):
    diag = Diagnostics(mag_data_opt.mag_vec, cost, max_iter=2000)
    for c in (0, 1, 2):
        print 'calc. diagnostics for {}-comp. at pos: {}'.format({0: 'x', 1: 'y', 2: 'z'}[c], pos)
        diag.pos = (c,) + pos
        gain_maps = diag.get_gain_row_maps()
        axis, cbar = gain_maps[0].display_phase('Gain map for pos: {}'.format(pos))
        axis.add_patch(Rectangle((diag.pos[3], diag.pos[2]), 1, 1, linewidth=2,
                                 color='g', fill=False))
        cbar.set_label(u'magnetization/phase [1/rad]', fontsize=15)
        diag.get_avg_kern_row().quiver_plot3d()
        mcon = diag.measure_contribution
        print 'measurement contr. (min - max): {:.2f} - {:.2f}'.format(mcon.min(), mcon.max())
        px_avrg, fwhm, (x, y, z) = diag.calculate_averaging()
        print 'px_avrg:', px_avrg
        print 'fwhm:', fwhm


diag_pos = [(0, 31, 31), (0, 16, 31)]


## SIMULATIONS:
#
#a = 10.
#b_0 = 1.
#lam = 1E-4
#dim = (1, 64, 64)
#center = (0.5, dim[1]//2-0.5, dim[2]//2-0.5)
#r = dim[1]//4
#
#
## SIMULATED VORTEX:
#
#mag_shape = mc.Shapes.disc(dim, center, radius=r, height=2)
#mag_vort = MagData(a, mc.create_mag_dist_vortex(mag_shape))
#mag_vort.quiver_plot('Original distribution')
#pm(mag_vort).display_combined('Original distribution')
#
## without mask:
#data_vort_nomask = DataSet(a, dim, b_0, mask=None)
#data_vort_nomask.projectors = [SimpleProjector(dim)]
#data_vort_nomask.phase_maps = data_vort_nomask.create_phase_maps(mag_vort)
#reg = FirstOrderRegularisator(data_vort_nomask.mask, lam, p=2)
#with TakeTime('reconstruction'):
#    mag_vort_nomask, cost_vort_nomask = rc.optimize_linear(data_vort_nomask, reg, max_iter=100)
#mag_vort_nomask.quiver_plot('Reconstructed distribution (without mask)')
#pm(mag_vort_nomask).display_combined('Reconstructed distribution (without mask)')
#pm(mag_vort_nomask-mag_vort).display_phase('Difference (without mask)')
##for pos in diag_pos:
##    calc_diagnostics(pos, mag_vort_nomask, cost_vort_nomask)
#
## with mask:
#data_vort_mask = DataSet(a, dim, b_0, mask=mag_vort.get_mask())
#data_vort_mask.projectors = [SimpleProjector(dim)]
#data_vort_mask.phase_maps = data_vort_mask.create_phase_maps(mag_vort)
#reg = FirstOrderRegularisator(data_vort_mask.mask, lam, p=2)
#with TakeTime('reconstruction'):
#    mag_vort_mask, cost_vort_mask = rc.optimize_linear(data_vort_mask, reg, max_iter=100)
#mag_vort_mask.quiver_plot('Reconstructed distribution (with mask)')
#pm(mag_vort_mask).display_combined('Reconstructed distribution (with mask)')
#pm(mag_vort_mask-mag_vort).display_phase('Difference (with mask)')
#
#
## SIMULATED VORTEX:
#
#mag_shape = magcreator.Shapes.slab(dim, center, width=(2, 2*r, 2*r))
#mag_slab = MagData(a, magcreator.create_mag_dist_homog(mag_shape, phi=np.pi/4))
#mag_slab.quiver_plot('Original distribution')
#pm(mag_slab).display_combined('Original distribution')
#
## without mask:
#data_slab_nomask = DataSet(a, dim, b_0, mask=None)
#data_slab_nomask.projectors = [SimpleProjector(dim)]
#data_slab_nomask.phase_maps = data_slab_nomask.create_phase_maps(mag_slab)
#reg = FirstOrderRegularisator(data_slab_nomask.mask, lam, p=2)
#with TakeTime('reconstruction'):
#    mag_slab_nomask, cost_slab_nomask = rc.optimize_linear(data_slab_nomask, reg, max_iter=100)
#mag_slab_nomask.quiver_plot('Reconstructed distribution (without mask)')
#pm(mag_slab_nomask).display_combined('Reconstructed distribution (without mask)')
#pm(mag_slab_nomask-mag_slab).display_phase('Difference (without mask)')
#
## with mask:
#data_slab_mask = DataSet(a, dim, b_0, mask=mag_slab.get_mask())
#data_slab_mask.projectors = [SimpleProjector(dim)]
#data_slab_mask.phase_maps = data_slab_mask.create_phase_maps(mag_slab)
#reg = FirstOrderRegularisator(data_slab_mask.mask, lam, p=2)
#with TakeTime('reconstruction'):
#    mag_slab_mask, cost_slab_mask = rc.optimize_linear(data_slab_mask, reg, max_iter=100)
#mag_slab_mask.quiver_plot('Reconstructed distribution (with mask)')
#pm(mag_slab_mask).display_combined('Reconstructed distribution (with mask)')
#pm(mag_slab_mask-mag_slab).display_phase('Difference (with mask)')


# MAGNETITE CUBES

a = 1.0  # in nm
b_0 = 1
lam = 1E-4

# 2 particles:
phase_map_2 = PhaseMap.load_from_netcdf4('../../../output/joern/phase_map_2.nc')
phase_map_2.display_combined('Original distribution')
with open('../../../output/joern/mask_2.pickle', 'rb') as pf:
    mask_2 = pickle.load(pf)
dim_2 = mask_2.shape
data_2 = DataSet(a, dim_2, b_0, mask=mask_2)
data_2.projectors = [SimpleProjector(dim_2)]
data_2.phase_maps = [phase_map_2]
reg = FirstOrderRegularisator(mask_2, lam, p=2)
with TakeTime('reconstruction'):
    mag_2, cost_2 = rc.optimize_linear(data_2, reg, max_iter=100)
mag_2.quiver_plot('Reconstructed distribution')
pm(mag_2).display_combined('Reconstructed distribution')
(pm(mag_2)-phase_map_2).display_phase('Difference')


# 4 particles:
phase_map_4 = PhaseMap.load_from_netcdf4('../../../output/joern/phase_map_4.nc')
phase_map_4.display_combined('Original distribution')
with open('../../../output/joern/mask_4.pickle', 'rb') as pf:
    mask_4 = pickle.load(pf)
dim_4 = mask_4.shape
data_4 = DataSet(a, dim_4, b_0, mask=mask_4)
data_4.projectors = [SimpleProjector(dim_4)]
data_4.phase_maps = [phase_map_4]
reg = FirstOrderRegularisator(mask_4, lam, p=2)
with TakeTime('reconstruction'):
    mag_4, cost_4 = rc.optimize_linear(data_4, reg, max_iter=100)
mag_4.quiver_plot('Reconstructed distribution')
pm(mag_4).display_combined('Reconstructed distribution')
(pm(mag_4)-phase_map_4).display_phase('Difference')



# EXPERIMENTAL NANOWIRE:
a = 1.455  # in nm
gain = 50
b_0 = 1
lam = 1E-4
PATH = '../../../output/patrick/'
PHASE = 'Reza_30_uj_tube_M'
MASK = 'Reza_30_uj_tube_maskbygimp'
FORMAT = '.tif'
longFOV = False
longFOV_string = np.where(longFOV, 'longFOV', 'normalFOV')
IMAGENAME = '{}_{}_{}_'.format(MASK, PHASE, longFOV_string)
PHASE_MAX = 0.5  # -10°: 0.92, 30°: 7.68
PHASE_MIN = -0.5  # -10°: -16.85, 30°: -18.89
PHASE_DELTA = PHASE_MAX - PHASE_MIN
###################################################################################################

# Read in files:
im_mask = Image.open(PATH+MASK+FORMAT)
#im_mask.thumbnail((125, 175), Image.ANTIALIAS)
im_phase = Image.open(PATH+PHASE+FORMAT)
#im_phase.thumbnail((125, 125), Image.ANTIALIAS)

mask = np.array(np.array(im_mask)/255, dtype=bool)
dim_uv = mask.shape
phase = (np.array(im_phase)/255.-0.5) * PHASE_DELTA
pad = dim_uv[0] - phase.shape[0]#25
phase_pad = np.zeros(dim_uv)
phase_pad[pad:, :] = phase#[pad:-pad, pad:-pad] = phase

mask = np.expand_dims(mask, axis=0)
dim = mask.shape

phase_map = PhaseMap(a, phase)
phase_map.display_combined('Original distribution')
phase_map_pad = PhaseMap(a, phase_pad)

data_set = DataSet(a, dim, b_0, mask)
data_set.append(phase_map_pad, SimpleProjector(dim))

reg = FirstOrderRegularisator(mask, lam, p=2)
with TakeTime('reconstruction'):
    mag_data_rec, cost_rec = rc.optimize_linear(data_set, reg, max_iter=100)
mag_data_rec.quiver_plot('Reconstructed distribution')
pm(mag_data_rec).display_combined('Reconstructed distribution')
(pm(mag_data_rec)-phase_map).display_phase('Difference')
