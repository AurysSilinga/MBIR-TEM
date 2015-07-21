# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 13:03:30 2015

@author: Jan
"""


import pyramid as pr
import warnings
import logging.config


logging.config.fileConfig(pr.LOGGING_CONFIG, disable_existing_loggers=False)

#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
logging.disable = True
warnings.warn('TEST')
from mayavi import mlab
warnings.warn('TEST')
logging.disable = False

print mlab
