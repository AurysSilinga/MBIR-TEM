# -*- coding: utf-8 -*-
"""Package for the creation and reconstruction of magnetic distributions and resulting phase maps.

Modules
-------
magcreator
    Create simple magnetic distributions.
magdata
    Class for the storage of magnetization data.
projector
    Class for projecting given magnetization distributions.
kernel
    Class for the kernel matrix representing one magnetized pixel.
phasemapper
    Create magnetic and electric phase maps from magnetization data.
phasemap
    Class for the storage of phase data.
analytic
    Create phase maps for magnetic distributions with analytic solutions.
datacollection
    Class for collecting pairs of phase maps and corresponding projectors.
forwardmodel
    Class which represents a phase mapping strategy.
costfunction
    Class for the evaluation of the cost of a function.
optimizer
    Provides strategies for optimizing first guess magnetic distributions.
reconstructor
    Reconstruct magnetic distributions from given phasemaps.

Subpackages
-----------
numcore
    Provides fast numerical functions for core routines.

"""


import logging, logging.config
import os


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF)
