# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:06:38 2014

@author: Jan
"""

import logging

def test():
    # 'application' code
    logger = logging.getLogger(__name__)
    logger.debug('debug message')
    logger.info('info message')
    logger.warn('warn message')
    logger.error('error message')
    logger.critical('critical message')
    
test()