# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""Config file for the pyramid package."""

import os

__all__ = ['DIR_PACKAGE', 'DIR_FILES', 'LOGGING_CONFIG', 'NTHREADS']

DIR_PACKAGE = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DIR_FILES = os.path.abspath(os.path.join(DIR_PACKAGE, os.pardir, 'files'))
LOGGING_CONFIG = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logging.ini')

try:
    # noinspection PyUnresolvedReferences
    import multiprocessing
    NTHREADS = multiprocessing.cpu_count()
    del multiprocessing
except ImportError:
    NTHREADS = 1

del os
