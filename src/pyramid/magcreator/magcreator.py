# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
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

import logging

import numpy as np
from numpy import pi

__all__ = ['create_mag_dist_homog', 'create_mag_dist_vortex', 'create_mag_dist_source',
           'create_mag_dist_smooth_vortex', 'create_mag_dist_skyrmion']
_log = logging.getLogger(__name__)

# TODO: generalise for scalar data? rename to fieldcreator? have subclasses vector, scalar?


def create_mag_dist_homog(mag_shape, phi, theta=pi / 2):
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

    Returns
    -------
    amplitude : tuple (N=3) of :class:`~numpy.ndarray` (N=3)
        The magnetic distribution as a tuple of the 3 components in
        `x`-, `y`- and `z`-direction on the 3-dimensional grid.

    """
    _log.debug('Calling create_mag_dist_homog')
    dim = np.shape(mag_shape)
    assert len(dim) == 3, 'Magnetic shapes must describe 3-dimensional distributions!'
    z_mag = np.ones(dim) * np.cos(theta) * mag_shape
    y_mag = np.ones(dim) * np.sin(theta) * np.sin(phi) * mag_shape
    x_mag = np.ones(dim) * np.sin(theta) * np.cos(phi) * mag_shape
    return np.array([x_mag, y_mag, z_mag])


def create_mag_dist_vortex(mag_shape, center=None, axis='z'):
    """Create a 3-dimensional magnetic distribution of a homogeneous magnetized object.

    Parameters
    ----------
    mag_shape : :class:`~numpy.ndarray` (N=3)
        The magnetic shapes (see :mod:`.~shapes`` for examples).
    center : tuple (N=2 or N=3), optional
        The vortex center, given in 2D `(v, u)` or 3D `(z, y, x)`, where the perpendicular axis
        is is discarded. Is set to the center of the field of view if not specified. The vortex
        center has to be between two pixels.
    axis :  {'z', '-z', 'y', '-y', 'x', '-x'}, optional
        The orientation of the vortex axis. The default is 'z'. Negative values invert the vortex
        orientation.

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
    assert center is None or len(center) in {2, 3}, \
        'Vortex center has to be defined in 3D or 2D or not at all!'
    if center is None:
        center = (dim[1] / 2, dim[2] / 2)
    sign = -1 if '-' in axis else 1
    if axis in ('z', '-z'):
        if len(center) == 3:  # if a 3D-center is given, just take the x and y components
            center = (center[1], center[2])
        u = np.linspace(-center[1], dim[2] - 1 - center[1], dim[2]) + 0.5  # pixel center!
        v = np.linspace(-center[0], dim[1] - 1 - center[0], dim[1]) + 0.5  # pixel center!
        uu, vv = np.meshgrid(u, v)
        phi = np.expand_dims(np.arctan2(vv, uu) - pi / 2, axis=0)
        phi = np.tile(phi, (dim[0], 1, 1))
        z_mag = np.zeros(dim)
        y_mag = np.ones(dim) * -np.sin(phi) * mag_shape * sign
        x_mag = np.ones(dim) * -np.cos(phi) * mag_shape * sign
    elif axis in ('y', '-y'):
        if len(center) == 3:  # if a 3D-center is given, just take the x and z components
            center = (center[0], center[2])
        u = np.linspace(-center[1], dim[2] - 1 - center[1], dim[2]) + 0.5  # pixel center!
        v = np.linspace(-center[0], dim[0] - 1 - center[0], dim[0]) + 0.5  # pixel center!
        uu, vv = np.meshgrid(u, v)
        phi = np.expand_dims(np.arctan2(vv, uu) - pi / 2, axis=1)
        phi = np.tile(phi, (1, dim[1], 1))
        z_mag = np.ones(dim) * np.sin(phi) * mag_shape * sign
        y_mag = np.zeros(dim)
        x_mag = np.ones(dim) * np.cos(phi) * mag_shape * sign
    elif axis in ('x', '-x'):
        if len(center) == 3:  # if a 3D-center is given, just take the z and y components
            center = (center[0], center[1])
        u = np.linspace(-center[1], dim[1] - 1 - center[1], dim[1]) + 0.5  # pixel center!
        v = np.linspace(-center[0], dim[0] - 1 - center[0], dim[0]) + 0.5  # pixel center!
        uu, vv = np.meshgrid(u, v)
        phi = np.expand_dims(np.arctan2(vv, uu) - pi / 2, axis=2)
        phi = np.tile(phi, (1, 1, dim[2]))
        z_mag = np.ones(dim) * -np.sin(phi) * mag_shape * sign
        y_mag = np.ones(dim) * -np.cos(phi) * mag_shape * sign
        x_mag = np.zeros(dim)
    else:
        raise ValueError('{} is not a valid argument (use x, -x, y, -y, z or -z)'.format(axis))
    return np.array([x_mag, y_mag, z_mag])


def create_mag_dist_smooth_vortex(mag_shape, center=None, vort_r=None, core_r=0, axis='z'):
    # TODO: Bring functions here in better order?
    """Create a 3-dimensional magnetic distribution of a homogeneous magnetized object.

    Parameters
    ----------
    mag_shape : :class:`~numpy.ndarray` (N=3)
        The magnetic shapes (see :mod:`.~shapes`` for examples).
    center : tuple (N=2 or N=3), optional
        The vortex center, given in 2D `(v, u)` or 3D `(z, y, x)`, where the perpendicular axis
        is is discarded. Is set to the center of the field of view if not specified. The vortex
        center has to be between two pixels.
    axis :  {'z', '-z', 'y', '-y', 'x', '-x'}, optional
        The orientation of the vortex axis. The default is 'z'. Negative values invert the vortex
        orientation.

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

    def core(r):
        """Function describing the smooth vortex core."""
        r_clip = np.clip(r - core_r, a_min=0, a_max=None)
        # TODO: vort_r = None throws error in next line (divide by None not defined) > set default!
        return 1 - 2/np.pi * np.arcsin(np.tanh(np.pi*r_clip/vort_r))

    _log.debug('Calling create_mag_dist_vortex')
    dim = mag_shape.shape
    assert len(dim) == 3, 'Magnetic shapes must describe 3-dimensional distributions!'
    assert center is None or len(center) in {2, 3}, \
        'Vortex center has to be defined in 3D or 2D or not at all!'
    if center is None:
        center = (dim[1] / 2, dim[2] / 2)
    sign = -1 if '-' in axis else 1
    if axis in ('z', '-z'):
        if len(center) == 3:  # if a 3D-center is given, just take the x and y components
            center = (center[1], center[2])
        u = np.linspace(-center[1], dim[2] - 1 - center[1], dim[2]) + 0.5  # pixel center!
        v = np.linspace(-center[0], dim[1] - 1 - center[0], dim[1]) + 0.5  # pixel center!
        uu, vv = np.meshgrid(u, v)
        rr = np.hypot(uu, vv)[None, :, :]
        phi = np.expand_dims(np.arctan2(vv, uu) - pi / 2, axis=0)
        phi = np.tile(phi, (dim[0], 1, 1))
        z_mag = np.ones(dim) * mag_shape * sign * core(rr)
        y_mag = np.ones(dim) * -np.sin(phi) * mag_shape * sign * np.sqrt(1 - core(rr))
        x_mag = np.ones(dim) * -np.cos(phi) * mag_shape * sign * np.sqrt(1 - core(rr))
    elif axis in ('y', '-y'):
        if len(center) == 3:  # if a 3D-center is given, just take the x and z components
            center = (center[0], center[2])
        u = np.linspace(-center[1], dim[2] - 1 - center[1], dim[2]) + 0.5  # pixel center!
        v = np.linspace(-center[0], dim[0] - 1 - center[0], dim[0]) + 0.5  # pixel center!
        uu, vv = np.meshgrid(u, v)
        rr = np.hypot(uu, vv)[:, None, :]
        phi = np.expand_dims(np.arctan2(vv, uu) - pi / 2, axis=1)
        phi = np.tile(phi, (1, dim[1], 1))
        z_mag = np.ones(dim) * np.sin(phi) * mag_shape * sign * np.sqrt(1 - core(rr))
        y_mag = np.ones(dim) * mag_shape * sign * core(rr)
        x_mag = np.ones(dim) * np.cos(phi) * mag_shape * sign * np.sqrt(1 - core(rr))
    elif axis in ('x', '-x'):
        if len(center) == 3:  # if a 3D-center is given, just take the z and y components
            center = (center[0], center[1])
        u = np.linspace(-center[1], dim[1] - 1 - center[1], dim[1]) + 0.5  # pixel center!
        v = np.linspace(-center[0], dim[0] - 1 - center[0], dim[0]) + 0.5  # pixel center!
        uu, vv = np.meshgrid(u, v)
        rr = np.hypot(uu, vv)[:, :, None]
        phi = np.expand_dims(np.arctan2(vv, uu) - pi / 2, axis=2)
        phi = np.tile(phi, (1, 1, dim[2]))
        z_mag = np.ones(dim) * -np.sin(phi) * mag_shape * sign * np.sqrt(1 - core(rr))
        y_mag = np.ones(dim) * -np.cos(phi) * mag_shape * sign * np.sqrt(1 - core(rr))
        x_mag = np.ones(dim) * mag_shape * sign * core(rr)
    else:
        raise ValueError('{} is not a valid argument (use x, -x, y, -y, z or -z)'.format(axis))
    return np.array([x_mag, y_mag, z_mag])


def create_mag_dist_source(mag_shape, center=None, axis='z'):
    """Create a 3-dimensional magnetic distribution of a homogeneous magnetized object.

    Parameters
    ----------
    mag_shape : :class:`~numpy.ndarray` (N=3)
        The magnetic shapes (see :mod:`.~shapes`` for examples).
    center : tuple (N=2 or N=3), optional
        The source center, given in 2D `(v, u)` or 3D `(z, y, x)`, where the perpendicular axis
        is discarded. Is set to the center of the field of view if not specified.
        The source center has to be between two pixels.
    axis :  {'z', '-z', 'y', '-y', 'x', '-x'}, optional
        The orientation of the source axis. The default is 'z'. Negative values invert the source
        to a sink.

    Returns
    -------
    amplitude : tuple (N=3) of :class:`~numpy.ndarray` (N=3)
        The magnetic distribution as a tuple of the 3 components in
        `x`-, `y`- and `z`-direction on the 3-dimensional grid.

    Notes
    -----
        To avoid singularities, the source center should lie between the pixel centers (which
        reside at coordinates with _.5 at the end), i.e. integer values should be used as center
        coordinates (e.g. coordinate 1 lies between the first and the second pixel).

    """
    _log.debug('Calling create_mag_dist_vortex')
    dim = mag_shape.shape
    assert len(dim) == 3, 'Magnetic shapes must describe 3-dimensional distributions!'
    assert center is None or len(center) in {2, 3}, \
        'Vortex center has to be defined in 3D or 2D or not at all!'
    if center is None:
        center = (dim[1] / 2, dim[2] / 2)
    sign = -1 if '-' in axis else 1
    if axis in ('z', '-z'):
        if len(center) == 3:  # if a 3D-center is given, just take the x and y components
            center = (center[1], center[2])
        u = np.linspace(-center[1], dim[2] - 1 - center[1], dim[2]) + 0.5  # pixel center!
        v = np.linspace(-center[0], dim[1] - 1 - center[0], dim[1]) + 0.5  # pixel center!
        uu, vv = np.meshgrid(u, v)
        phi = np.expand_dims(np.arctan2(vv, uu) - pi / 2, axis=0)
        phi = np.tile(phi, (dim[0], 1, 1))
        z_mag = np.zeros(dim)
        y_mag = np.ones(dim) * np.cos(phi) * mag_shape * sign
        x_mag = np.ones(dim) * -np.sin(phi) * mag_shape * sign
    elif axis in ('y', '-y'):
        if len(center) == 3:  # if a 3D-center is given, just take the x and z components
            center = (center[0], center[2])
        u = np.linspace(-center[1], dim[2] - 1 - center[1], dim[2]) + 0.5  # pixel center!
        v = np.linspace(-center[0], dim[0] - 1 - center[0], dim[0]) + 0.5  # pixel center!
        uu, vv = np.meshgrid(u, v)
        phi = np.expand_dims(np.arctan2(vv, uu) - pi / 2, axis=1)
        phi = np.tile(phi, (1, dim[1], 1))
        z_mag = np.ones(dim) * np.cos(phi) * mag_shape * sign
        y_mag = np.zeros(dim)
        x_mag = np.ones(dim) * -np.sin(phi) * mag_shape * sign
    elif axis in ('x', '-x'):
        if len(center) == 3:  # if a 3D-center is given, just take the z and y components
            center = (center[0], center[1])
        u = np.linspace(-center[1], dim[1] - 1 - center[1], dim[1]) + 0.5  # pixel center!
        v = np.linspace(-center[0], dim[0] - 1 - center[0], dim[0]) + 0.5  # pixel center!
        uu, vv = np.meshgrid(u, v)
        phi = np.expand_dims(np.arctan2(vv, uu) - pi / 2, axis=2)
        phi = np.tile(phi, (1, 1, dim[2]))
        z_mag = np.ones(dim) * np.cos(phi) * mag_shape * sign
        y_mag = np.ones(dim) * -np.sin(phi) * mag_shape * sign
        x_mag = np.zeros(dim)
    else:
        raise ValueError('{} is not a valid argument (use x, -x, y, -y, z or -z)'.format(axis))
    return np.array([x_mag, y_mag, z_mag])


def create_mag_dist_skyrmion(mag_shape, center=None, phi_0=0, skyrm_d=None, wall_d=None,
                             axis='z', mode='1'):
    """Create a 3-dimensional magnetic Bloch or Neel type skyrmion distribution.

    Parameters
    ----------
    mag_shape : :class:`~numpy.ndarray` (N=3)
        The magnetic shapes (see :mod:`.~shapes`` for examples).
    center : tuple (N=2 or N=3), optional
        The source center, given in 2D `(v, u)` or 3D `(z, y, x)`, where the perpendicular axis
        is discarded. Is set to the center of the field of view if not specified.
        The center has to be between two pixels.
    phi_0 : float, optional
        Angular offset switching between Neel type (0 [default] or pi) or Bloch type (+/- pi/2)
        skyrmions.
    skyrm_d : float, optional
        Diameter of the skyrmion. Defaults to half of the smaller dimension perpendicular to the
        skyrmion axis if not specified.
    wall_d : float, optional
        Diameter of the domain wall of the skyrmion. Defaults to `skyrm_d / 4` if not specified.
    axis :  {'z', '-z', 'y', '-y', 'x', '-x'}, optional
        The orientation of the skyrmion axis. The default is 'z'. Negative values invert skyrmion
        core direction.

    Returns
    -------
    amplitude : tuple (N=3) of :class:`~numpy.ndarray` (N=3)
        The magnetic distribution as a tuple of the 3 components in
        `x`-, `y`- and `z`-direction on the 3-dimensional grid.

    Notes
    -----
        To avoid singularities, the source center should lie between the pixel centers (which
        reside at coordinates with _.5 at the end), i.e. integer values should be used as center
        coordinates (e.g. coordinate 1 lies between the first and the second pixel).

        Skyrmion wall width is dependant onexchange stiffness  A [J/m] and anisotropy K [J/m³]
        The out-of-plane magnetization at the domain wall can be described as:
        Mz = -Ms * tanh(x/w)
        w = sqrt(A/K)

    """

    def m_oop_tanh(r, wall_r, skyrm_r):

        return -np.tanh((r-skyrm_r)/wall_r)

    def m_oop_arctan_exp(r, wall_r, skyrm_r):
        theta = np.arctan(np.exp((r-skyrm_r)/wall_r))  # range: 0 to pi/2!
        y = 2 * 2*theta/np.pi - 1  # normalise (*2/np.pi), double range (*2), shift to range (-1, 1)
        return -y  # scale to new max_amp

    # Determine the calculation model for the out-of-plane component:
    _log.debug('Calling create_mag_dist_skyrmion')
    dim = mag_shape.shape
    assert len(dim) == 3, 'Magnetic shapes must describe 3-dimensional distributions!'
    assert center is None or len(center) in {2, 3}, \
        'Vortex center has to be defined in 3D or 2D or not at all!'
    assert mode in ('tanh', 'arctan_exp'), '`mode` can only be `tanh` or `arctan_exp`!'
    # Determine out-of-plane component model:
    if mode == 'tanh':
        m_oop = m_oop_tanh
    elif mode == 'arctan_exp':
        m_oop = m_oop_arctan_exp
    if center is None:
        center = (dim[1] / 2, dim[2] / 2)
    sign = -1 if '-' in axis else 1
    if axis in ('z', '-z'):
        if len(center) == 3:  # if a 3D-center is given, just take the x and y components
            center = (center[1], center[2])
        if skyrm_d is None:
            skyrm_d = np.min((dim[1], dim[2])) / 2
        if wall_d is None:
            wall_d = skyrm_d / 4
        u = np.linspace(-center[1], dim[2] - 1 - center[1], dim[2]) + 0.5  # pixel center!
        v = np.linspace(-center[0], dim[1] - 1 - center[0], dim[1]) + 0.5  # pixel center!
        uu, vv = np.meshgrid(u, v)
        rr = np.hypot(uu, vv)
        # Out-of-plane component:
        z_mag = sign * m_oop(rr, wall_d/2, skyrm_d/2)
        # In-plane component:
        r_mag = 1 - np.abs(z_mag)
        # Separate in-plane into x/y-components:
        phi = np.arctan2(vv, uu) - phi_0
        y_mag = r_mag * np.sin(phi)
        x_mag = r_mag * np.cos(phi)
        # Extend to 3D and take shape into account:  # TODO: Einheitliche Extension to 3D überall!
        z_mag = np.repeat(z_mag[None, :, :], dim[0], axis=0) * mag_shape
        y_mag = np.repeat(y_mag[None, :, :], dim[0], axis=0) * mag_shape
        x_mag = np.repeat(x_mag[None, :, :], dim[0], axis=0) * mag_shape
    elif axis in ('y', '-y'):
        if len(center) == 3:  # if a 3D-center is given, just take the x and z components
            center = (center[0], center[2])
        if skyrm_d is None:
            skyrm_d = np.min((dim[0], dim[2])) / 2
        if wall_d is None:
            wall_d = skyrm_d / 4
        u = np.linspace(-center[1], dim[2] - 1 - center[1], dim[2]) + 0.5  # pixel center!
        v = np.linspace(-center[0], dim[0] - 1 - center[0], dim[0]) + 0.5  # pixel center!
        uu, vv = np.meshgrid(u, v)
        rr = np.hypot(uu, vv)
        # Out-of-plane component:
        y_mag = sign * m_oop(rr, wall_d/2, skyrm_d/2)
        # In-plane component:
        r_mag = 1 - np.abs(y_mag)
        # Separate in-plane into x/y-components:
        phi = np.arctan2(vv, uu) - phi_0
        x_mag = r_mag * np.cos(phi)
        z_mag = r_mag * np.sin(phi)
        # Extend to 3D and take shape into account:  # TODO: Einheitliche Extension to 3D überall!
        z_mag = np.repeat(z_mag[:, None, :], dim[1], axis=1) * mag_shape
        y_mag = np.repeat(y_mag[:, None, :], dim[1], axis=1) * mag_shape
        x_mag = np.repeat(x_mag[:, None, :], dim[1], axis=1) * mag_shape
    elif axis in ('x', '-x'):
        if len(center) == 3:  # if a 3D-center is given, just take the z and y components
            center = (center[0], center[1])
        if skyrm_d is None:
            skyrm_d = np.min((dim[0], dim[1])) / 2
        if wall_d is None:
            wall_d = skyrm_d / 4
        u = np.linspace(-center[1], dim[1] - 1 - center[1], dim[1]) + 0.5  # pixel center!
        v = np.linspace(-center[0], dim[0] - 1 - center[0], dim[0]) + 0.5  # pixel center!
        uu, vv = np.meshgrid(u, v)
        rr = np.hypot(uu, vv)
        # Out-of-plane component:
        x_mag = sign * m_oop(rr, wall_d/2, skyrm_d/2)
        # In-plane component:
        r_mag = 1 - np.abs(x_mag)
        # Separate in-plane into x/y-components:
        phi = np.arctan2(vv, uu) - phi_0
        z_mag = r_mag * np.sin(phi)
        y_mag = r_mag * np.cos(phi)
        # Extend to 3D and take shape into account:  # TODO: Einheitliche Extension to 3D überall!
        z_mag = np.repeat(z_mag[:, :, None], dim[2], axis=2) * mag_shape
        y_mag = np.repeat(y_mag[:, :, None], dim[2], axis=2) * mag_shape
        x_mag = np.repeat(x_mag[:, :, None], dim[2], axis=2) * mag_shape
    else:
        raise ValueError('{} is not a valid argument (use x, -x, y, -y, z or -z)'.format(axis))
    return np.array([x_mag, y_mag, z_mag])
