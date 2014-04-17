# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:06:42 2014

@author: Jan
"""


import numpy as np
import vtk
#import netCDF4
import logging
import sys

import pickle

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)

reader = vtk.vtkDataSetReader()
reader.SetFileName('irect_500x125x3.vtk')
reader.ReadAllScalarsOn()
reader.ReadAllVectorsOn()
reader.Update()

output = reader.GetOutput()
size = output.GetNumberOfPoints()

vtk_points = output.GetPoints().GetData()
vtk_vectors = output.GetPointData().GetVectors()

point_array = np.zeros(vtk_points.GetSize())
vtk_points.ExportToVoidPointer(point_array)
point_array = np.reshape(point_array, (-1, 3))

vector_array = np.zeros(vtk_points.GetSize())
vtk_vectors.ExportToVoidPointer(vector_array)
vector_array = np.reshape(vector_array, (-1, 3))

data = np.hstack((point_array, vector_array))

log.info('Data reading complete!')

#magfile = netCDF4.Dataset('tube_90x30x30.nc', 'w', format='NETCDF3_64BIT')
#magfile.createDimension('comp', 6)  # Number of components
#magfile.createDimension('size', size)
#
#x = magfile.createVariable('x', 'f8', ('size'))
#y = magfile.createVariable('y', 'f8', ('size'))
#z = magfile.createVariable('z', 'f8', ('size'))
#x_mag = magfile.createVariable('x_mag', 'f8', ('size'))
#y_mag = magfile.createVariable('y_mag', 'f8', ('size'))
#z_mag = magfile.createVariable('z_mag', 'f8', ('size'))
#
#log.info('Start saving data separately!')
#x = data[:, 0]
#y = data[:, 1]
#z = data[:, 2]
#x_mag = data[:, 3]
#y_mag = data[:, 4]
#z_mag = data[:, 5]
#log.info('Separate saving complete!')
#
#log.info('Try saving the whole array!')
#filedata = magfile.createVariable('data', 'f8', ('size', 'comp'))
#filedata[:, :] = data
#log.info('Saving complete!')
#
#magfile.close()

log.info('Pickling data!')
with open('vtk_to_numpy.pickle', 'w') as pf:
    pickle.dump(data, pf)
log.info('Pickling complete!')
