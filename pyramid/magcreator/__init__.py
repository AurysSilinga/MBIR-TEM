# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#

from . import shapes
from . import examples
from .magcreator import *

import logging
_log = logging.getLogger(__name__)
del logging


__all__ = ['shapes', 'examples']
__all__.extend(magcreator.__all__)
