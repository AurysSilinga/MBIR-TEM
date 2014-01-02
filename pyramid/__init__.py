# -*- coding: utf-8 -*-
"""Package for the creation and reconstruction of magnetic distributions and resulting phase maps.

Modules
-------
magcreator
    Create simple magnetic distributions.
magdata
    Class for the storage of magnetizatin data.
projector
    Create projections of a given magnetization distribution.
kernel
    Class for the kernel matrix representing one magnetized pixel.
phasemapper
    Create magnetic and electric phase maps from magnetization data.
phasemap
    Class for the storage of phase data.
analytic
    Create phase maps for magnetic distributions with analytic solutions.
holoimage
    Create holographic contour maps from a given phase map.
reconstructor
    Reconstruct magnetic distributions from given phasemaps.

Subpackages
-----------
numcore
    Provides fast numerical functions for core routines.

"""


import logging, logging.config
import os

LOGGING_CONF = os.path.join(os.path.dirname(__file__), 'logging.ini')

logging.config.fileConfig(LOGGING_CONF)


log = logging.getLogger(__name__)
log.info('imported package, log:'+log.name)