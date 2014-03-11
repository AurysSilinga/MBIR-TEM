# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:06:01 2014

@author: Jan
"""


from pyramid.magdata import MagData
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PMAdapterFM, PMConvolve, PMFourier, PMReal

from time import clock

#import cProfile


mag_data = MagData.load_from_netcdf4('../output/vtk data/tube_90x30x30.nc')

projector = SimpleProjector(mag_data.dim)

start = clock()
pm_adapter = PMAdapterFM(mag_data.a, projector)
pm_convolve = PMConvolve(mag_data.a, projector)
pm_fourier = PMFourier(mag_data.a, projector, padding=3)
pm_real = PMReal(mag_data.a, projector)
print 'Overhead  :', clock()-start

#cProfile.run('phasemapper(mag_data)')#, filename='../output/profile.profile')

start = clock()
pm_adapter(mag_data)
print 'Adapter FM:', clock()-start
start = clock()
pm_convolve(mag_data)
print 'Convolve  :', clock()-start
start = clock()
pm_fourier(mag_data)
print 'Fourier   :', clock()-start
start = clock()
pm_real(mag_data)
print 'Real      :', clock()-start

phase_map = pm_convolve(mag_data)

(-phase_map).display_combined(density=16)





##from xml.etree import ElementTree
#from cProfile import Profile
##xml_content = '<a>\n' + '\t<b/><c><d>text</d></c>\n' * 100 + '</a>'
#profiler = Profile()
##profiler.runctx("ElementTree.fromstring(xml_content)", locals(), globals())
#profiler.run('phasemapper(mag_data)')
#
#import pyprof2calltree
#from pyprof2calltree import convert, visualize
#
#pyprof2calltree.visualize(profiler.getstats())                            # run kcachegrind
#pyprof2calltree.convert(profiler.getstats(), 'profiling_results.kgrind')  # save for later