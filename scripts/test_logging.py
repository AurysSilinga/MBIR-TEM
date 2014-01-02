# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:58:32 2014

@author: Jan
"""

##from test_logging_sub import test
#
#import logging
#
### create logger
##logger = logging.getLogger(__name__+'2')
##logger.setLevel(logging.DEBUG)
##
### create console handler and set level to debug
##ch = logging.StreamHandler()
##ch.setLevel(logging.DEBUG)
##
### create formatter
##formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
##
### add formatter to ch
##ch.setFormatter(formatter)
##
### add ch to logger
##logger.addHandler(ch)
##
### 'application' code
##logger.debug('debug message')
##logger.info('info message')
##logger.warn('warn message')
##logger.error('error message')
##logger.critical('critical message')
#
##test()
#
#
#
#
#
#
#import logging.config
#
#logging.config.fileConfig('logging.conf')
#
## create logger
#logger = logging.getLogger('simpleExample')
#
## 'application' code
#logger.debug('debug message')
#logger.info('info message')
#logger.warn('warn message')
#logger.error('error message')
#logger.critical('critical message')











#import logging
#
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)
#
## create a file handler
#handler = logging.FileHandler('hello.log', 'w')
#handler.setLevel(logging.INFO)
#
## create a logging format
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#handler.setFormatter(formatter)
#
## add the handlers to the logger
#logger.addHandler(handler)
#
#logger.info('Start reading database')
## read database here
#records = {'john': 55, 'tom': 66}
#logger.debug('Records: %s', records)
#logger.info('Updating records ...')
## update records here
#logger.info('Finish updating records')





import loggingtest.logmodule as lm

lm.test()






