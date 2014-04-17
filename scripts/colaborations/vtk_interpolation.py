# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:09:08 2014

@author: Jan
"""

from pylab import *
import pickle
from tqdm import tqdm
from pyramid.magdata import MagData

with open("vtk_to_numpy.pickle") as pf:
    data = pickle.load(pf)
zs =  unique(data[:,2])

axis = plt.figure().add_subplot(1, 1, 1)
axis.scatter(data[:, 0], data[:, 1])
plt.show()

# # regular grid
xs = linspace(-8.5*2, 8.5*2, 256)
ys = linspace(-9.5*2, 9.5*2, 256)

o, p = meshgrid(xs, ys)

newdata = zeros((len(xs), len(ys), len(zs), data.shape[1] - 3))


## WITH MASKING OF THE CENTER:
#
#iz_x = concatenate([linspace(-4.95, -4.95, 50),
#                    linspace(-4.95, 0, 50),
#                    linspace(0, 4.95, 50),
#                    linspace(4.95, 4.95, 50),
#                    linspace(-4.95, 0, 50),
#                    linspace(0, 4.95, 50),])
#iz_y = concatenate([linspace(-2.88,  2.88, 50),
#                    linspace(2.88,  5.7, 50),
#                    linspace(5.7,  2.88, 50),
#                    linspace(2.88, -2.88, 50),
#                    linspace(-2.88,  -5.7, 50),
#                    linspace(-5.7,  -2.88, 50), ])
#
#
#for i, z in tqdm(enumerate(zs), total=len(zs)):
#    subdata = data[data[:, 2] == z, :]
#
#    for j in range(newdata.shape[-1]):
#        gridded_subdata = griddata(concatenate([subdata[:, 0], iz_x]),
#        concatenate([subdata[:, 1], iz_y]), concatenate([subdata[:, 3 + j],
#        zeros(len(iz_x))]), o, p)
#        newdata[:, :, i, j] = gridded_subdata.filled(fill_value=0)


# WITHOUT MASKING OF THE CENTER:


for i, z in tqdm(enumerate(zs), total=len(zs)):
    subdata = data[data[:, 2] == z, :]

    for j in range(3):  # For all 3 components!
        gridded_subdata = griddata(subdata[:, 0], subdata[:, 1], subdata[:, 3 + j], o, p)
        newdata[:, :, i, j] = gridded_subdata.filled(fill_value=0)


magnitude = newdata.swapaxes(0,3).swapaxes(1,2).swapaxes(2,3)

mag_data = MagData(1., magnitude)

mag_data.quiver_plot()

mag_data.save_to_netcdf4('vtk_mag_data.nc')
