# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""Create simple magnetic distributions.

The :mod:`~.magcreator` module is responsible for the creation of simple distributions of
magnetic moments. In the :class:`~.Shapes` class, you can find several general shapes for the
3-dimensional volume that should be magnetized (e.g. slabs, spheres, discs or single pixels).
These magnetic shapes are then used as input for the creating functions (or you could specify the
volume yourself as a 3-dimensional boolean matrix or a matrix with values in the range from 0 to 1,
which modifies the magnetization amplitude). The specified volume can either be magnetized
homogeneously with the :func:`~.create_mag_dist_homog` function by specifying the magnetization
direction, or in a vortex state with the :func:`~.create_mag_dist_vortex` by specifying the vortex
center.

"""

import numpy as np
from numpy import pi

import abc

import logging


__all__ = ['Shapes', 'create_mag_dist_homog', 'create_mag_dist_vortex']
_log = logging.getLogger(__name__)


class Shapes(object):

    '''Abstract class containing functions for generating magnetic shapes.

    The :class:`~.Shapes` class is a collection of some methods that return a 3-dimensional
    matrix that represents the magnetized volume and consists of values between 0 and 1.
    This matrix is used in the functions of the :mod:`~.magcreator` module to create
    :class:`~pyramid.magdata.MagData` objects which store the magnetic informations.

    '''

    __metaclass__ = abc.ABCMeta
    _log = logging.getLogger(__name__+'.Shapes')

    @classmethod
    def slab(cls, dim, center, width):
        '''Create the shape of a slab.

        Parameters
        ----------
        dim : tuple (N=3)
            The dimensions of the grid `(z, y, x)`.
        center : tuple (N=3)
            The center of the slab in pixel coordinates `(z, y, x)`.
        width : tuple (N=3)
            The width of the slab in pixel coordinates `(z, y, x)`.

        Returns
        -------
        mag_shape : :class:`~numpy.ndarray` (N=3)
            The magnetic shape as a 3D-array with values between 1 and 0.

        '''
        cls._log.debug('Calling slab')
        assert np.shape(dim) == (3,), 'Parameter dim has to be a tuple of length 3!'
        assert np.shape(center) == (3,), 'Parameter center has to be a tuple of length 3!'
        assert np.shape(width) == (3,), 'Parameter width has to be a tuple of length 3!'
        mag_shape = np.array([[[abs(x - center[2]) <= width[2] / 2
                             and abs(y - center[1]) <= width[1] / 2
                             and abs(z - center[0]) <= width[0] / 2
                             for x in range(dim[2])]
                             for y in range(dim[1])]
                             for z in range(dim[0])])
        return mag_shape

    @classmethod
    def disc(cls, dim, center, radius, height, axis='z'):
        '''Create the shape of a cylindrical disc in x-, y-, or z-direction.

        Parameters
        ----------
        dim : tuple (N=3)
            The dimensions of the grid `(z, y, x)`.
        center : tuple (N=3)
            The center of the disc in pixel coordinates `(z, y, x)`.
        radius : float
            The radius of the disc in pixel coordinates.
        height : float
            The height of the disc in pixel coordinates.
        axis : {'z', 'y', 'x'}, optional
            The orientation of the disc axis. The default is 'z'.

        Returns
        -------
        mag_shape : :class:`~numpy.ndarray` (N=3)
            The magnetic shape as a 3D-array with values between 1 and 0.

        '''
        cls._log.debug('Calling disc')
        assert np.shape(dim) == (3,), 'Parameter dim has to be a a tuple of length 3!'
        assert np.shape(center) == (3,), 'Parameter center has to be a a tuple of length 3!'
        assert radius > 0 and np.shape(radius) == (), 'Radius has to be a positive scalar value!'
        assert height > 0 and np.shape(height) == (), 'Height has to be a positive scalar value!'
        assert axis in {'z', 'y', 'x'}, 'Axis has to be x, y or z (as a string)!'
        if axis == 'z':
            mag_shape = np.array([[[np.hypot(x - center[2], y - center[1]) <= radius
                                 and abs(z - center[0]) <= height / 2
                                 for x in range(dim[2])]
                                 for y in range(dim[1])]
                                 for z in range(dim[0])])
        elif axis == 'y':
            mag_shape = np.array([[[np.hypot(x - center[2], z - center[0]) <= radius
                                 and abs(y - center[1]) <= height / 2
                                 for x in range(dim[2])]
                                 for y in range(dim[1])]
                                 for z in range(dim[0])])
        elif axis == 'x':
            mag_shape = np.array([[[np.hypot(y - center[1], z - center[0]) <= radius
                                 and abs(x - center[2]) <= height / 2
                                 for x in range(dim[2])]
                                 for y in range(dim[1])]
                                 for z in range(dim[0])])
        return mag_shape

    @classmethod
    def ellipse(cls, dim, center, width, height, axis='z'):
        '''Create the shape of an elliptical cylinder in x-, y-, or z-direction.

        Parameters
        ----------
        dim : tuple (N=3)
            The dimensions of the grid `(z, y, x)`.
        center : tuple (N=3)
            The center of the ellipse in pixel coordinates `(z, y, x)`.
        width : tuple (N=2)
            Length of the two axes of the ellipse.
        height : float
            The height of the ellipse in pixel coordinates.
        axis : {'z', 'y', 'x'}, optional
            The orientation of the ellipse axis. The default is 'z'.

        Returns
        -------
        mag_shape : :class:`~numpy.ndarray` (N=3)
            The magnetic shape as a 3D-array with values between 1 and 0.

        '''
        cls._log.debug('Calling ellipse')
        assert np.shape(dim) == (3,), 'Parameter dim has to be a a tuple of length 3!'
        assert np.shape(center) == (3,), 'Parameter center has to be a a tuple of length 3!'
        assert np.shape(width) == (2,), 'Parameter width has to be a a tuple of length 2!'
        assert height > 0 and np.shape(height) == (), 'Height has to be a positive scalar value!'
        assert axis in {'z', 'y', 'x'}, 'Axis has to be x, y or z (as a string)!'
        if axis == 'z':
            mag_shape = np.array([[[np.hypot((x-center[2])/(width[1]/2.),
                                             (y-center[1])/(width[0]/2.)) <= 1
                                 and abs(z - center[0]) <= height / 2
                                 for x in range(dim[2])]
                                 for y in range(dim[1])]
                                 for z in range(dim[0])])
        elif axis == 'y':
            mag_shape = np.array([[[np.hypot((x-center[2])/(width[1]/2.),
                                             (z-center[0])/(width[0]/2.)) <= 1
                                 and abs(y-center[1]) <= height / 2
                                 for x in range(dim[2])]
                                 for y in range(dim[1])]
                                 for z in range(dim[0])])
        elif axis == 'x':
            mag_shape = np.array([[[np.hypot((y-center[1])/(width[1]/2.),
                                             (z-center[0])/(width[0]/2.)) <= 1
                                 and abs(x - center[2]) <= height / 2
                                 for x in range(dim[2])]
                                 for y in range(dim[1])]
                                 for z in range(dim[0])])
        return mag_shape

    @classmethod
    def sphere(cls, dim, center, radius):
        '''Create the shape of a sphere.

        Parameters
        ----------
        dim : tuple (N=3)
            The dimensions of the grid `(z, y, x)`.
        center : tuple (N=3)
            The center of the sphere in pixel coordinates `(z, y, x)`.
        radius : float
            The radius of the sphere in pixel coordinates.

        Returns
        -------
        mag_shape : :class:`~numpy.ndarray` (N=3)
            The magnetic shape as a 3D-array with values between 1 and 0.

        '''
        cls._log.debug('Calling sphere')
        assert np.shape(dim) == (3,), 'Parameter dim has to be a a tuple of length 3!'
        assert np.shape(center) == (3,), 'Parameter center has to be a a tuple of length 3!'
        assert radius > 0 and np.shape(radius) == (), 'Radius has to be a positive scalar value!'
        mag_shape = np.array([[[np.sqrt((x-center[2])**2
                                        + (y-center[1])**2
                                        + (z-center[0])**2) <= radius
                             for x in range(dim[2])]
                             for y in range(dim[1])]
                             for z in range(dim[0])])
        return mag_shape

    @classmethod
    def ellipsoid(cls, dim, center, width):
        '''Create the shape of an ellipsoid.

        Parameters
        ----------
        dim : tuple (N=3)
            The dimensions of the grid `(z, y, x)`.
        center : tuple (N=3)
            The center of the ellipsoid in pixel coordinates `(z, y, x)`.
        width : tuple (N=3)
            The width of the ellipsoid `(z, y, x)`.

        Returns
        -------
        mag_shape : :class:`~numpy.ndarray` (N=3)
            The magnetic shape as a 3D-array with values between 1 and 0.

        '''
        cls._log.debug('Calling ellipsoid')
        assert np.shape(dim) == (3,), 'Parameter dim has to be a a tuple of length 3!'
        assert np.shape(center) == (3,), 'Parameter center has to be a a tuple of length 3!'
        assert np.shape(width) == (3,), 'Parameter width has to be a a tuple of length 3!'
        mag_shape = np.array([[[(x-center[2])**2/(width[2]/2)**2
                                + (y-center[1])**2/(width[1]/2)**2
                                + (z-center[0])**2/(width[0]/2)**2 <= 1
                             for x in range(dim[2])]
                             for y in range(dim[1])]
                             for z in range(dim[0])])
        return mag_shape

    @classmethod
    def filament(cls, dim, pos, axis='y'):
        '''Create the shape of a filament.

        Parameters
        ----------
        dim : tuple (N=3)
            The dimensions of the grid `(z, y, x)`.
        pos : tuple (N=2)
            The position of the filament in pixel coordinates `(coord1, coord2)`.
            `coord1` and `coord2` stand for the two axes, which are perpendicular to `axis`. For
            the default case (`axis = y`), it is `(coord1, coord2) = (z, x)`.
        axis :  {'y', 'x', 'z'}, optional
            The orientation of the filament axis. The default is 'y'.

        Returns
        -------
        mag_shape : :class:`~numpy.ndarray` (N=3)
            The magnetic shape as a 3D-array with values between 1 and 0.

        '''
        cls._log.debug('Calling filament')
        assert np.shape(dim) == (3,), 'Parameter dim has to be a tuple of length 3!'
        assert np.shape(pos) == (2,), 'Parameter pos has to be a tuple of length 2!'
        assert axis in {'z', 'y', 'x'}, 'Axis has to be x, y or z (as a string)!'
        mag_shape = np.zeros(dim)
        if axis == 'z':
            mag_shape[:, pos[0], pos[1]] = 1
        elif axis == 'y':
            mag_shape[pos[0], :, pos[1]] = 1
        elif axis == 'x':
            mag_shape[pos[0], pos[1], :] = 1
        return mag_shape

    @classmethod
    def pixel(cls, dim, pixel):
        '''Create the shape of a single pixel.

        Parameters
        ----------
        dim : tuple (N=3)
            The dimensions of the grid `(z, y, x)`.
        pixel : tuple (N=3)
            The coordinates of the pixel `(z, y, x)`.

        Returns
        -------
        mag_shape : :class:`~numpy.ndarray` (N=3)
            The magnetic shape as a 3D-array with values between 1 and 0.

        '''
        cls._log.debug('Calling pixel')
        assert np.shape(dim) == (3,), 'Parameter dim has to be a tuple of length 3!'
        assert np.shape(pixel) == (3,), 'Parameter pixel has to be a tuple of length 3!'
        mag_shape = np.zeros(dim)
        mag_shape[pixel] = 1
        return mag_shape


def create_mag_dist_homog(mag_shape, phi, theta=pi/2, magnitude=1):
    '''Create a 3-dimensional magnetic distribution of a homogeneously magnetized object.

    Parameters
    ----------
    mag_shape : :class:`~numpy.ndarray` (N=3)
        The magnetic shapes (see Shapes.* for examples).
    phi : float
        The azimuthal angle, describing the direction of the magnetized object.
    theta : float, optional
        The polar angle, describing the direction of the magnetized object.
        The default is pi/2, which means, the z-component is zero.
    magnitude : float, optional
        The relative magnitude for the magnetic shape. The default is 1.

    Returns
    -------
    magnitude : tuple (N=3) of :class:`~numpy.ndarray` (N=3)
        The magnetic distribution as a tuple of the 3 components in
        `x`-, `y`- and `z`-direction on the 3-dimensional grid.

    '''
    _log.debug('Calling create_mag_dist_homog')
    dim = np.shape(mag_shape)
    assert len(dim) == 3, 'Magnetic shapes must describe 3-dimensional distributions!'
    z_mag = np.ones(dim) * np.cos(theta) * mag_shape * magnitude
    y_mag = np.ones(dim) * np.sin(theta) * np.sin(phi) * mag_shape * magnitude
    x_mag = np.ones(dim) * np.sin(theta) * np.cos(phi) * mag_shape * magnitude
    return np.array([x_mag, y_mag, z_mag])


def create_mag_dist_vortex(mag_shape, center=None, axis='z', magnitude=1):
    '''Create a 3-dimensional magnetic distribution of a homogeneous magnetized object.

    Parameters
    ----------
    mag_shape : :class:`~numpy.ndarray` (N=3)
        The magnetic shapes (see :class:`.~Shapes` for examples).
    center : tuple (N=2 or N=3), optional
        The vortex center, given in 2D `(v, u)` or 3D `(z, y, x)`, where the perpendicular axis
        is is discarded. Is set to the center of the field of view if not specified. The vortex
        center has to be between two pixels.
    magnitude : float, optional
        The relative magnitude for the magnetic shape. The default is 1. Negative signs reverse
        the vortex direction (left-handed instead of right-handed).
    axis :  {'z', 'y', 'x'}, optional
        The orientation of the vortex axis. The default is 'z'.
    Returns
    -------
    magnitude : tuple (N=3) of :class:`~numpy.ndarray` (N=3)
        The magnetic distribution as a tuple of the 3 components in
        `x`-, `y`- and `z`-direction on the 3-dimensional grid.

    '''
    _log.debug('Calling create_mag_dist_vortex')
    dim = np.shape(mag_shape)
    assert len(dim) == 3, 'Magnetic shapes must describe 3-dimensional distributions!'
    assert axis in {'z', 'y', 'x'}, 'Axis has to be x, y or z (as a string)!'
    assert center is None or len(center) in {2, 3}, \
        'Vortex center has to be defined in 3D or 2D or not at all!'
    if center is None:
        center = (int(dim[1]/2)-0.5, int(dim[2]/2)-0.5)  # center has to be between (!) two pixels
    if axis == 'z':
        if len(center) == 3:  # if a 3D-center is given, just take the x and y components
            center = (center[1], center[2])
        u = np.linspace(-center[1], dim[2]-1-center[1], dim[2])
        v = np.linspace(-center[0], dim[1]-1-center[0], dim[1])
        uu, vv = np.meshgrid(u, v)
        phi = np.expand_dims(np.arctan2(vv, uu)-pi/2, axis=0)
        phi = np.tile(phi, (dim[0], 1, 1))
        z_mag = np.zeros(dim)
        y_mag = -np.ones(dim) * np.sin(phi) * mag_shape * magnitude
        x_mag = -np.ones(dim) * np.cos(phi) * mag_shape * magnitude
    elif axis == 'y':
        if len(center) == 3:  # if a 3D-center is given, just take the x and z components
            center = (center[0], center[2])
        u = np.linspace(-center[1], dim[2]-1-center[1], dim[2])
        v = np.linspace(-center[0], dim[0]-1-center[0], dim[0])
        uu, vv = np.meshgrid(u, v)
        phi = np.expand_dims(np.arctan2(vv, uu)-pi/2, axis=1)
        phi = np.tile(phi, (1, dim[1], 1))
        z_mag = np.ones(dim) * np.sin(phi) * mag_shape * magnitude
        y_mag = np.zeros(dim)
        x_mag = np.ones(dim) * np.cos(phi) * mag_shape * magnitude
    elif axis == 'x':
        if len(center) == 3:  # if a 3D-center is given, just take the z and y components
            center = (center[0], center[1])
        u = np.linspace(-center[1], dim[1]-1-center[1], dim[1])
        v = np.linspace(-center[0], dim[0]-1-center[0], dim[0])
        uu, vv = np.meshgrid(u, v)
        phi = np.expand_dims(np.arctan2(vv, uu)-pi/2, axis=2)
        phi = np.tile(phi, (1, 1, dim[2]))
        z_mag = -np.ones(dim) * np.sin(phi) * mag_shape * magnitude
        y_mag = -np.ones(dim) * np.cos(phi) * mag_shape * magnitude
        x_mag = np.zeros(dim)
    return np.array([x_mag, y_mag, z_mag])
