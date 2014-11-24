# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 14:17:56 2014

@author: Jan
"""

from PIL import Image
import numpy as np

from jutil.taketime import TakeTime
from pyramid import *

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)

###################################################################################################
a = 1.0  # in nm
gain = 5
b_0 = 1
lam = 1E-6
PATH = '../../output/patrick/'
dim_uv = (128, 128)
###################################################################################################

# Read in files:
im_mask = Image.open(PATH+'min102mask.tif')
im_mask.thumbnail(dim_uv, Image.ANTIALIAS)
mask = np.array(np.array(im_mask)/255, dtype=bool)
mask = np.expand_dims(mask, axis=0)
im_phase = Image.open(PATH+'min102.tif')
im_phase.thumbnail(dim_uv, Image.ANTIALIAS)
phase = np.array(im_phase)/255. - 0.5#*(18.19977+8.057761) - 8.057761
phase_map = PhaseMap(a, phase)
dim = mask.shape

data_set = DataSet(a, dim, b_0, mask)
data_set.append(phase_map, SimpleProjector(dim))

regularisator = FirstOrderRegularisator(mask, lam, p=2)

with TakeTime('reconstruction time'):
    mag_data_rec = rc.optimize_linear(data_set, regularisator=regularisator, max_iter=50)

# Display the reconstructed phase map and holography image:
phase_map.display_combined('Input PhaseMap', gain=40)
mag_data_rec.quiver_plot(log=True)
phase_map_rec = pm(mag_data_rec)
phase_map_rec.display_combined('Reconstr. Distribution', gain=40)
#plt.savefig(dirname + "/reconstr.png")
