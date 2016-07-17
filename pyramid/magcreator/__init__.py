# -*- coding: utf-8 -*-

from .magcreator import *
from . import shapes
from . import examples

import logging
_log = logging.getLogger(__name__)
del logging

__all__ = ['shapes', 'examples']
__all__.extend(magcreator.__all__)
