# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:34:24 2015

@author: Jan
"""


import numpy as np
import pyramid as py
from mayavi import mlab


def plot3d(x, y, z, values):
    mlab.figure()
    plot = mlab.points3d(x, y, z, values, opacity=0.5)
    mlab.outline(plot)
    mlab.axes(plot)
    return plot


dim = (4, 4, 4)
mag_3d = np.zeros(dim)
mag_3d[1:-1, 1:-1, 1:-1] = 1
zz, yy, xx = np.indices(dim)
z, x, y = zz.flatten(), yy.flatten(), xx.flatten()
plot = plot3d(x, y, z, mag_3d.flatten())


projector_z = py.SimpleProjector(dim, axis='z')
projector_y = py.SimpleProjector(dim, axis='y')
projector_x = py.SimpleProjector(dim, axis='x')
projection_z = projector_z.weight.dot(mag_3d.flatten()).reshape(dim[1:])
projection_y = projector_y.weight.dot(mag_3d.flatten()).reshape(dim[1:])
projection_x = projector_x.weight.dot(mag_3d.flatten()).reshape(dim[1:])

mask_z = np.where(projection_z > 0, True, False)
mask_y = np.where(projection_y > 0, True, False)
mask_x = np.where(projection_x > 0, True, False)
extrusion_z = projector_z.weight.T.dot(mask_z.flatten())
extrusion_y = projector_y.weight.T.dot(mask_y.flatten())
extrusion_x = projector_x.weight.T.dot(mask_x.flatten())
plot_z = plot3d(x, y, z, extrusion_z)
plot_y = plot3d(x, y, z, extrusion_y)
plot_x = plot3d(x, y, z, extrusion_x)

extrusion_list = np.asarray([extrusion_z, extrusion_y, extrusion_x])
extrusion_sum = extrusion_list.sum(axis=0)
plot_mask = plot3d(x, y, z, extrusion_sum)


mask_z_inv = np.where(projection_z > 0, False, True)
mask_y_inv = np.where(projection_y > 0, False, True)
mask_x_inv = np.where(projection_x > 0, False, True)
extrusion_z_inv = projector_z.weight.T.dot(mask_z_inv.flatten())
extrusion_y_inv = projector_y.weight.T.dot(mask_y_inv.flatten())
extrusion_x_inv = projector_x.weight.T.dot(mask_x_inv.flatten())

extrusion_inv_list = np.asarray([extrusion_z_inv, extrusion_y_inv, extrusion_x_inv])
extrusion_inv_sum = extrusion_inv_list.sum(axis=0)
plot_mask = plot3d(x, y, z, extrusion_inv_sum)

mask_3d = np.where(extrusion_inv_sum == 0, True, False)
plot_mask = plot3d(x, y, z, mask_3d.astype(dtype=np.int))

mlab.figure()
plot_contour = plot = mlab.points3d(x, y, z, mask_3d.astype(dtype=np.int), opacity=0.5,
                                    mode='cube', scale_factor=1)
mlab.outline(plot_contour)
mlab.axes(plot_contour)
