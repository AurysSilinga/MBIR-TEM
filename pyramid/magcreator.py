# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""Create simple magnetic distributions.

The :mod:`~.magcreator` module is responsible for the creation of simple distributions of
magnetic moments. In the :mod:`~.shapes` module, you can find several general shapes for the
3-dimensional volume that should be magnetized (e.g. slabs, spheres, discs or single pixels).
These shapes are then used as input for the creating functions (or you could specify the
volume yourself as a 3-dimensional boolean matrix or a matrix with values in the range from 0 to 1,
which modifies the magnetization amplitude). The specified volume can either be magnetized
homogeneously with the :func:`~.create_mag_dist_homog` function by specifying the magnetization
direction, or in a vortex state with the :func:`~.create_mag_dist_vortex` by specifying the vortex
center.

"""

from __future__ import division

import logging

import numpy as np
from numpy import pi

__all__ = ['create_mag_dist_homog', 'create_mag_dist_vortex']
_log = logging.getLogger(__name__)


def create_mag_dist_homog(mag_shape, phi, theta=pi / 2, amplitude=1):
    """Create a 3-dimensional magnetic distribution of a homogeneously magnetized object.

    Parameters
    ----------
    mag_shape : :class:`~numpy.ndarray` (N=3)
        The magnetic shapes (see :mod:`.~shapes` for examples).
    phi : float
        The azimuthal angle, describing the direction of the magnetized object.
    theta : float, optional
        The polar angle, describing the direction of the magnetized object.
        The default is pi/2, which means, the z-component is zero.
    amplitude : float, optional
        The relative amplitude for the magnetic shape. The default is 1.

    Returns
    -------
    amplitude : tuple (N=3) of :class:`~numpy.ndarray` (N=3)
        The magnetic distribution as a tuple of the 3 components in
        `x`-, `y`- and `z`-direction on the 3-dimensional grid.

    """
    _log.debug('Calling create_mag_dist_homog')
    dim = np.shape(mag_shape)
    assert len(dim) == 3, 'Magnetic shapes must describe 3-dimensional distributions!'
    z_mag = np.ones(dim) * np.cos(theta) * mag_shape * amplitude
    y_mag = np.ones(dim) * np.sin(theta) * np.sin(phi) * mag_shape * amplitude
    x_mag = np.ones(dim) * np.sin(theta) * np.cos(phi) * mag_shape * amplitude
    return np.array([x_mag, y_mag, z_mag])


def create_mag_dist_vortex(mag_shape, center=None, axis='z', amplitude=1):
    """Create a 3-dimensional magnetic distribution of a homogeneous magnetized object.

    Parameters
    ----------
    mag_shape : :class:`~numpy.ndarray` (N=3)
        The magnetic shapes (see :mod:`.~shapes`` for examples).
    center : tuple (N=2 or N=3), optional
        The vortex center, given in 2D `(v, u)` or 3D `(z, y, x)`, where the perpendicular axis
        is is discarded. Is set to the center of the field of view if not specified. The vortex
        center has to be between two pixels.
    amplitude : float, optional
        The relative amplitude for the magnetic shape. The default is 1. Negative signs reverse
        the vortex direction (left-handed instead of right-handed).
    axis :  {'z', 'y', 'x'}, optional
        The orientation of the vortex axis. The default is 'z'.

    Returns
    -------
    amplitude : tuple (N=3) of :class:`~numpy.ndarray` (N=3)
        The magnetic distribution as a tuple of the 3 components in
        `x`-, `y`- and `z`-direction on the 3-dimensional grid.

    Notes
    -----
        To avoid singularities, the vortex center should lie between the pixel centers (which
        reside at coordinates with _.5 at the end), i.e. integer values should be used as center
        coordinates (e.g. coordinate 1 lies between the first and the second pixel).

    """
    _log.debug('Calling create_mag_dist_vortex')
    dim = mag_shape.shape
    assert len(dim) == 3, 'Magnetic shapes must describe 3-dimensional distributions!'
    assert axis in {'z', 'y', 'x'}, 'Axis has to be x, y or z (as a string)!'
    assert center is None or len(center) in {2, 3}, \
        'Vortex center has to be defined in 3D or 2D or not at all!'
    if center is None:
        center = (dim[1] / 2, dim[2] / 2)
    if axis == 'z':
        if len(center) == 3:  # if a 3D-center is given, just take the x and y components
            center = (center[1], center[2])
        u = np.linspace(-center[1], dim[2] - 1 - center[1], dim[2]) + 0.5  # pixel center!
        v = np.linspace(-center[0], dim[1] - 1 - center[0], dim[1]) + 0.5  # pixel center!
        uu, vv = np.meshgrid(u, v)
        phi = np.expand_dims(np.arctan2(vv, uu) - pi / 2, axis=0)
        phi = np.tile(phi, (dim[0], 1, 1))
        z_mag = np.zeros(dim)
        y_mag = -np.ones(dim) * np.sin(phi) * mag_shape * amplitude
        x_mag = -np.ones(dim) * np.cos(phi) * mag_shape * amplitude
    elif axis == 'y':
        if len(center) == 3:  # if a 3D-center is given, just take the x and z components
            center = (center[0], center[2])
        u = np.linspace(-center[1], dim[2] - 1 - center[1], dim[2]) + 0.5  # pixel center!
        v = np.linspace(-center[0], dim[0] - 1 - center[0], dim[0]) + 0.5  # pixel center!
        uu, vv = np.meshgrid(u, v)
        phi = np.expand_dims(np.arctan2(vv, uu) - pi / 2, axis=1)
        phi = np.tile(phi, (1, dim[1], 1))
        z_mag = np.ones(dim) * np.sin(phi) * mag_shape * amplitude
        y_mag = np.zeros(dim)
        x_mag = np.ones(dim) * np.cos(phi) * mag_shape * amplitude
    elif axis == 'x':
        if len(center) == 3:  # if a 3D-center is given, just take the z and y components
            center = (center[0], center[1])
        u = np.linspace(-center[1], dim[1] - 1 - center[1], dim[1]) + 0.5  # pixel center!
        v = np.linspace(-center[0], dim[0] - 1 - center[0], dim[0]) + 0.5  # pixel center!
        uu, vv = np.meshgrid(u, v)
        phi = np.expand_dims(np.arctan2(vv, uu) - pi / 2, axis=2)
        phi = np.tile(phi, (1, 1, dim[2]))
        z_mag = -np.ones(dim) * np.sin(phi) * mag_shape * amplitude
        y_mag = -np.ones(dim) * np.cos(phi) * mag_shape * amplitude
        x_mag = np.zeros(dim)
    else:
        raise ValueError('{} is not a valid argument (use x, y or z)'.format(axis))
    return np.array([x_mag, y_mag, z_mag])
