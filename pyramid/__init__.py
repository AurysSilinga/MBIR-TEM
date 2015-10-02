# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
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
dataset
    Class for collecting pairs of phase maps and corresponding projectors.
forwardmodel
    Class which represents a phase mapping strategy.
costfunction
    Class for the evaluation of the cost of a function.
reconstruction
    Reconstruct magnetic distributions from given phasemaps.
regularisator
    Class to instantiate different regularisation strategies.
ramp
    Class which is used to add polynomial ramps to phasemaps.
diagnostics
    Class to calculate diagnostics
quaternion
    Class which is used for easy rotations in the Projector classes.
colormap
    Class which implements a custom direction encoding colormap.
fft
    Class for custom FFT functions using numpy or FFTW.

Subpackages
-----------
numcore
    Provides fast numerical functions for core routines.

"""


from . import analytic
from . import magcreator
from . import reconstruction
from . import fft
from .costfunction import *  # analysis:ignore
from .dataset import *  # analysis:ignore
from .diagnostics import *  # analysis:ignore
from .forwardmodel import *  # analysis:ignore
from .kernel import *  # analysis:ignore
from .magdata import *  # analysis:ignore
from .phasemap import *  # analysis:ignore
from .phasemapper import *  # analysis:ignore
from .projector import *  # analysis:ignore
from .regularisator import *  # analysis:ignore
from .ramp import *  # analysis:ignore
from .quaternion import *  # analysis:ignore
from .colormap import *  # analysis:ignore
from .config import *  # analysis:ignore
from .version import version as __version__
from .version import hg_revision as __hg_revision__

import logging
_log = logging.getLogger(__name__)
_log.info("Starting Pyramid V{} HG{}".format(__version__, __hg_revision__))
del logging

__all__ = ['analytic', 'magcreator', 'reconstruction', 'fft']
__all__.extend(costfunction.__all__)
__all__.extend(dataset.__all__)
__all__.extend(diagnostics.__all__)
__all__.extend(forwardmodel.__all__)
__all__.extend(kernel.__all__)
__all__.extend(magdata.__all__)
__all__.extend(phasemap.__all__)
__all__.extend(phasemapper.__all__)
__all__.extend(projector.__all__)
__all__.extend(regularisator.__all__)
__all__.extend(ramp.__all__)
__all__.extend(quaternion.__all__)
__all__.extend(colormap.__all__)
