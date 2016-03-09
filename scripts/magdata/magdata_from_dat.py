# -*- coding: utf-8 -*-
"""Create magnetization distributions from fortran sorted txt-files."""


import os
import numpy as np
import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
filename = 'J=1.D=0.084.H=0.0067.Bobber.dat'
scale = 1
###################################################################################################


#def func():
path = os.path.join(py.DIR_FILES, 'dat', filename)
data = np.genfromtxt(path, dtype=np.float32, delimiter=',', usecols=(0, 1, 2, 3, 4, 5))
x, y, z, xmag, ymag, zmag = data.T
a = (y[1] - y[0]) * scale
dim_z = len(np.unique(z))
dim_y = len(np.unique(y))
dim_x = len(np.unique(x))
dim = (dim_z, dim_x, dim_y)  # Order of descending variance!
xmag = xmag.reshape(dim).swapaxes(1, 2)
ymag = ymag.reshape(dim).swapaxes(1, 2)
zmag = zmag.reshape(dim).swapaxes(1, 2)
magnitude = np.array((xmag, ymag, zmag))
mag_data = py.VectorData(a, magnitude)
#mag_data.save_to_netcdf4('magdata_dat_{}'.format(filename.replace('.dat', '.nc')))
mag_data.quiver_plot3d(ar_dens=4, coloring='amplitude')
mag_data.quiver_plot3d(ar_dens=4, coloring='angle')
py.pm(mag_data).display_combined(interpolation='bilinear')


#def funci():
#    path = os.path.join(py.DIR_FILES, 'dat', filename)
#    with open(path) as f:
#        import csv
#        reader = csv.reader(f)
#        x, y, z, xmag, ymag, zmag = [], [], [], [], [], []
#        for row in reader:
#            x.append(int(row[0]))
#            y.append(int(row[1]))
#            z.append(int(row[2]))
#            xmag.append(float(row[3]))
#            ymag.append(float(row[4]))
#            zmag.append(float(row[5]))
#    a = (y[1] - y[0]) * scale
#    dim_z = len(np.unique(z))
#    dim_y = len(np.unique(y))
#    dim_x = len(np.unique(x))
#    dim = (dim_z, dim_x, dim_y)  # Order of descending variance!
#    xmag = np.reshape(xmag, dim).swapaxes(1, 2)
#    ymag = np.reshape(ymag, dim).swapaxes(1, 2)
#    zmag = np.reshape(zmag, dim).swapaxes(1, 2)
#    field = np.array((xmag, ymag, zmag))
#    mag_data = py.VectorData(a, field)
#    return mag_data
##   mag_data.save_to_netcdf4('magdata_dat_{}'.format(filename.replace('.dat', '.nc')))
#
#if __name__ == '__main__':
#    mag_data = funci()
#
#
