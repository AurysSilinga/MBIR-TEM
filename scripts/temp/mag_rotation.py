# -*- coding: utf-8 -*-
"""
Created on Mon Mar 02 13:47:00 2015

@author: Jan
"""


import numpy as np
import pyramid as py


mag_shape = py.magcreator.Shapes.slab((5, 6, 7), (1, 1.5, 2), (3, 4, 5))
magnitude = py.magcreator.create_mag_dist_homog(mag_shape, np.pi/4, np.pi/4)
py.MagData(1, magnitude).quiver_plot3d()
dim = magnitude.shape[1:]

# flip x:
magnitude_flipx = magnitude[:, :, :, ::-1]
mag_x, mag_y, mag_z = magnitude_flipx
py.MagData(1, np.array((-mag_x, mag_y, mag_z))).quiver_plot3d()

# flip y:
magnitude_flipy = magnitude[:, :, ::-1, :]
mag_x, mag_y, mag_z = magnitude_flipy
py.MagData(1, np.array((mag_x, -mag_y, mag_z))).quiver_plot3d()

# flip z:
magnitude_flipz = magnitude[:, ::-1, :, :]
mag_x, mag_y, mag_z = magnitude_flipz
py.MagData(1, np.array((mag_x, mag_y, -mag_z))).quiver_plot3d()

# x-rotation:
magnitude_rot = np.zeros((3, dim[1], dim[0], dim[2]))
for i in range(dim[2]):
    mag_x, mag_y, mag_z = magnitude[:, :, :, i]
    mag_xrot, mag_yrot, mag_zrot = np.rot90(mag_x, 3), np.rot90(mag_y, 3), np.rot90(mag_y, 3)
    magnitude_rot[:, :, :, i] = np.array((mag_xrot, -mag_zrot, mag_yrot))
py.MagData(1, magnitude_rot).quiver_plot3d()

# y-rotation:
magnitude_rot = np.zeros((3, dim[2], dim[1], dim[0]))
for i in range(dim[1]):
    mag_x, mag_y, mag_z = magnitude[:, :, i, :]
    mag_xrot, mag_yrot, mag_zrot = np.rot90(mag_x), np.rot90(mag_y), np.rot90(mag_y)
    magnitude_rot[:, :, i, :] = np.array((mag_zrot, mag_yrot, -mag_xrot))
py.MagData(1, magnitude_rot).quiver_plot3d()

# z-rotation:
magnitude_rot = np.zeros((3, dim[0], dim[2], dim[1]))
for i in range(dim[0]):
    mag_x, mag_y, mag_z = magnitude[:, i, :, :]
    mag_xrot, mag_yrot, mag_zrot = np.rot90(mag_x, 3), np.rot90(mag_y, 3), np.rot90(mag_y, 3)
    magnitude_rot[:, i, :, :] = np.array((-mag_yrot, mag_xrot, mag_zrot))
py.MagData(1, magnitude_rot).quiver_plot3d()
