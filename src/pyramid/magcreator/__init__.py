# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""Subpackage containing functionality for creating magnetic distributions."""

from . import shapes
from . import examples
from .magcreator import *

__all__ = ['shapes', 'examples']
__all__.extend(magcreator.__all__)
