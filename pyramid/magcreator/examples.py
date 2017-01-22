# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""Provide simple examples for magnetic distributions."""

import logging

import numpy as np

import random as rnd

from . import magcreator as mc
from . import shapes
from ..fielddata import VectorData


__all__ = ['pyramid_logo', 'singularity', 'homog_pixel', 'homog_slab', 'homog_disc',
           'homog_sphere', 'homog_filament', 'homog_alternating_filament',
           'homog_array_sphere_disc_slab', 'homog_random_pixels', 'homog_random_slabs',
           'vortex_slab', 'vortex_disc', 'vortex_alternating_discs', 'vortex_sphere',
           'vortex_horseshoe', 'smooth_vortex_disc', 'source_disc',
           'core_shell_disc', 'core_shell_sphere']
_log = logging.getLogger(__name__)


def pyramid_logo(a=1., dim=(1, 256, 256), phi=np.pi / 2, theta=np.pi / 2):
    """Create pyramid logo."""
    _log.debug('Calling pyramid_logo')
    mag_shape = np.zeros(dim)
    x = range(dim[2])
    y = range(dim[1])
    xx, yy = np.meshgrid(x, y)
    bottom = (yy >= 0.25 * dim[1])
    left = (yy <= 0.75 / 0.5 * dim[1] / dim[2] * xx)
    right = np.fliplr(left)
    mag_shape[0, ...] = np.logical_and(np.logical_and(left, right), bottom)
    return VectorData(a, mc.create_mag_dist_homog(mag_shape, phi, theta))


def singularity(a=1., dim=(8, 8, 8), center=None):
    """Create magnetic singularity."""
    _log.debug('Calling singularity')
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    x = np.linspace(-center[2], dim[2] - 1 - center[2], dim[2]) + 0.5  # pixel center!
    y = np.linspace(-center[1], dim[1] - 1 - center[1], dim[1]) + 0.5  # pixel center!
    z = np.linspace(-center[0], dim[0] - 1 - center[0], dim[0]) + 0.5  # pixel center!
    yy, zz, xx = np.meshgrid(x, y, z)  # What's up with this strange order???
    magnitude = np.array((xx, yy, zz)).astype(float)
    magnitude /= np.sqrt((magnitude ** 2 + 1E-30).sum(axis=0))  # Normalise!
    return VectorData(a, magnitude)


def homog_pixel(a=1., dim=(1, 9, 9), pixel=None, phi=np.pi/4, theta=np.pi/2):
    """Create single magnetised slab."""
    _log.debug('Calling homog_pixel')
    if pixel is None:
        pixel = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    mag_shape = shapes.pixel(dim, pixel)
    return VectorData(a, mc.create_mag_dist_homog(mag_shape, phi, theta))


def homog_slab(a=1., dim=(32, 32, 32), center=None, width=None, phi=np.pi/4, theta=np.pi/4):
    """Create homogeneous slab magnetisation distribution."""
    _log.debug('Calling homog_slab')
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if width is None:
        width = (np.max((dim[0] // 8, 1)), np.max((dim[1] // 2, 1)), np.max((dim[2] // 4, 1)))
    mag_shape = shapes.slab(dim, center, width)
    return VectorData(a, mc.create_mag_dist_homog(mag_shape, phi, theta))


def homog_disc(a=1., dim=(32, 32, 32), center=None, radius=None, height=None,
               phi=np.pi / 4, theta=np.pi / 4):
    """Create homogeneous disc magnetisation distribution."""
    _log.debug('Calling homog_disc')
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if radius is None:
        radius = dim[2] // 4
    if height is None:
        height = np.max((dim[0] // 2, 1))
    mag_shape = shapes.disc(dim, center, radius, height)
    return VectorData(a, mc.create_mag_dist_homog(mag_shape, phi, theta))


def homog_sphere(a=1., dim=(32, 32, 32), center=None, radius=None, phi=np.pi/4, theta=np.pi/4):
    """Create homogeneous sphere magnetisation distribution."""
    _log.debug('Calling homog_sphere')
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if radius is None:
        radius = dim[2] // 4
    mag_shape = shapes.sphere(dim, center, radius)
    return VectorData(a, mc.create_mag_dist_homog(mag_shape, phi, theta))


def homog_filament(a=1., dim=(1, 21, 21), pos=None, phi=np.pi / 2, theta=np.pi/2):
    """Create magnetisation distribution of a single magnetised filaments."""
    _log.debug('Calling homog_filament')
    if pos is None:
        pos = (0, dim[1] // 2)
    mag_shape = shapes.filament(dim, pos)
    return VectorData(a, mc.create_mag_dist_homog(mag_shape, phi, theta))


def homog_alternating_filament(a=1., dim=(1, 21, 21), spacing=5, phi=np.pi/2, theta=np.pi/2):
    """Create magnetisation distribution of alternating filaments."""
    _log.debug('Calling homog_alternating_filament')
    count = int((dim[1] - 1) / spacing) + 1
    magdata = VectorData(a, np.zeros((3,) + dim))
    for i in range(count):
        pos = i * spacing
        mag_shape = shapes.filament(dim, (0, pos))
        magdata += VectorData(a, mc.create_mag_dist_homog(mag_shape, phi, theta))
        phi *= -1  # Switch the angle
    return magdata


def homog_array_sphere_disc_slab(a=1., dim=(64, 128, 128), center_sp=(32, 96, 64), radius_sp=24,
                                 center_di=(32, 32, 96), radius_di=24, height_di=24,
                                 center_sl=(32, 32, 32), width_sl=(48, 48, 48)):
    """Create array of several  magnetisation distribution (sphere, disc and slab)."""
    _log.debug('Calling homog_array_sphere_disc_slab')
    mag_shape_sphere = shapes.sphere(dim, center_sp, radius_sp)
    mag_shape_disc = shapes.disc(dim, center_di, radius_di, height_di)
    mag_shape_slab = shapes.slab(dim, center_sl, width_sl)
    magdata = VectorData(a, mc.create_mag_dist_homog(mag_shape_sphere, np.pi))
    magdata += VectorData(a, mc.create_mag_dist_homog(mag_shape_disc, np.pi / 2))
    magdata += VectorData(a, mc.create_mag_dist_homog(mag_shape_slab, np.pi / 4))
    return magdata


def homog_random_pixels(a=1., dim=(1, 64, 64), count=10, rnd_seed=24):
    """Create random magnetised pixels."""
    _log.debug('Calling homog_random_pixels')
    rnd.seed(rnd_seed)
    magdata = VectorData(a, np.zeros((3,) + dim))
    for i in range(count):
        pixel = (rnd.randrange(dim[0]), rnd.randrange(dim[1]), rnd.randrange(dim[2]))
        mag_shape = shapes.pixel(dim, pixel)
        phi = 2 * np.pi * rnd.random()
        magdata += VectorData(a, mc.create_mag_dist_homog(mag_shape, phi))
    return magdata


def homog_random_slabs(a=1., dim=(1, 64, 64), count=10, width_max=5, rnd_seed=2):
    """Create random magnetised slabs."""
    _log.debug('Create homog_random_slabs')
    rnd.seed(rnd_seed)
    magdata = VectorData(a, np.zeros((3,) + dim))
    for i in range(count):
        width = (1, rnd.randint(1, width_max), rnd.randint(1, width_max))
        center = (rnd.randrange(int(width[0] / 2), dim[0] - int(width[0] / 2)),
                  rnd.randrange(int(width[1] / 2), dim[1] - int(width[1] / 2)),
                  rnd.randrange(int(width[2] / 2), dim[2] - int(width[2] / 2)))
        mag_shape = shapes.slab(dim, center, width)
        phi = 2 * np.pi * rnd.random()
        magdata += VectorData(a, mc.create_mag_dist_homog(mag_shape, phi))
    return magdata


def vortex_slab(a=1., dim=(32, 32, 32), center=None, width=None, axis='z'):
    """Create vortex slab magnetisation distribution."""
    _log.debug('Calling vortex_slab')
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if width is None:
        width = (np.max((dim[0] // 2, 1)), np.max((dim[1] // 2, 1)), np.max((dim[2] // 2, 1)))
    mag_shape = shapes.slab(dim, center, width)
    magnitude = mc.create_mag_dist_vortex(mag_shape, center, axis)
    return VectorData(a, magnitude)


def vortex_disc(a=1., dim=(32, 32, 32), center=None, radius=None, height=None, axis='z'):
    """Create vortex disc magnetisation distribution."""
    _log.debug('Calling vortex_disc')
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if radius is None:
        radius = dim[2] // 4
    if height is None:
        height = np.max((dim[0] // 2, 1))
    mag_shape = shapes.disc(dim, center, radius, height, axis)
    magnitude = mc.create_mag_dist_vortex(mag_shape, center, axis)
    return VectorData(a, magnitude)


def vortex_alternating_discs(a=1., dim=(80, 32, 32), count=8):
    """Create pillar of alternating vortex disc magnetisation distributions."""
    _log.debug('Calling vortex_alternating_discs')
    segment_height = dim[0] // (count + 2)
    magdata = VectorData(a, np.zeros((3,) + dim))
    for i in range(count):
        axis = 'z' if i % 2 == 0 else '-z'
        center = (segment_height * (i + 1 + 0.5), dim[1] // 2, dim[2] // 2)
        radius = dim[2] // 4
        height = segment_height
        mag_shape = shapes.disc(dim, center=center, radius=radius, height=height)
        mag_amp = mc.create_mag_dist_vortex(mag_shape=mag_shape, center=center, axis=axis)
        magdata += VectorData(1, mag_amp)
    return magdata


def vortex_sphere(a=1., dim=(32, 32, 32), center=None, radius=None, axis='z'):
    """Create vortex sphere magnetisation distribution."""
    _log.debug('Calling vortex_sphere')
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if radius is None:
        radius = dim[2] // 4
    mag_shape = shapes.sphere(dim, center, radius)
    magnitude = mc.create_mag_dist_vortex(mag_shape, center, axis)
    return VectorData(a, magnitude)


def vortex_horseshoe(a=1., dim=(16, 64, 64), center=None, radius_core=None,
                     radius_shell=None, height=None):
    """Create magnetic horseshoe vortex magnetisation distribution."""
    _log.debug('Calling vortex_horseshoe')
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if radius_core is None:
        radius_core = dim[1] // 8
    if radius_shell is None:
        radius_shell = dim[1] // 4
    if height is None:
        height = np.max((dim[0] // 2, 1))
    mag_shape_core = shapes.disc(dim, center, radius_core, height)
    mag_shape_outer = shapes.disc(dim, center, radius_shell, height)
    mag_shape_horseshoe = np.logical_xor(mag_shape_outer, mag_shape_core)
    mag_shape_horseshoe[:, dim[1] // 2:, :] = False
    return VectorData(a, mc.create_mag_dist_vortex(mag_shape_horseshoe))


def smooth_vortex_disc(a=1., dim=(32, 32, 32), center=None, radius=None, height=None, axis='z',
                       vortex_radius=None):
    """Create smooth vortex disc magnetisation distribution."""
    _log.debug('Calling vortex_disc')
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if radius is None:
        radius = dim[2] // 4
    if height is None:
        height = np.max((dim[0] // 2, 1))
    if vortex_radius is None:
        vortex_radius = radius // 2
    mag_shape = shapes.disc(dim, center, radius, height, axis)
    magnitude = mc.create_mag_dist_smooth_vortex(mag_shape, center, vortex_radius, axis)
    return VectorData(a, magnitude)


def source_disc(a=1., dim=(32, 32, 32), center=None, radius=None, height=None, axis='z'):
    """Create source disc magnetisation distribution."""
    _log.debug('Calling vortex_disc')
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if radius is None:
        radius = dim[2] // 4
    if height is None:
        height = np.max((dim[0] // 2, 1))
    mag_shape = shapes.disc(dim, center, radius, height, axis)
    magnitude = mc.create_mag_dist_source(mag_shape, center, axis)
    return VectorData(a, magnitude)


def core_shell_disc(a=1., dim=(32, 32, 32), center=None, radius_core=None,
                    radius_shell=None, height=None, rate_core_to_shell=0.75):
    """Create magnetic core shell disc magnetisation distribution."""
    _log.debug('Calling core_shell_disc')
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if radius_core is None:
        radius_core = dim[1] // 8
    if radius_shell is None:
        radius_shell = dim[1] // 4
    if height is None:
        height = np.max((dim[0] // 2, 1))
    mag_shape_core = shapes.disc(dim, center, radius_core, height)
    mag_shape_outer = shapes.disc(dim, center, radius_shell, height)
    mag_shape_shell = np.logical_xor(mag_shape_outer, mag_shape_core)
    magdata = VectorData(a, mc.create_mag_dist_vortex(mag_shape_shell)) * rate_core_to_shell
    magdata += VectorData(a, mc.create_mag_dist_homog(mag_shape_core, phi=0, theta=0))
    return magdata


def core_shell_sphere(a=1., dim=(32, 32, 32), center=None, radius_core=None,
                      radius_shell=None, rate_core_to_shell=0.75):
    """Create magnetic core shell sphere magnetisation distribution."""
    _log.debug('Calling core_shell_sphere')
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if radius_core is None:
        radius_core = dim[1] // 8
    if radius_shell is None:
        radius_shell = dim[1] // 4
    mag_shape_sphere = shapes.sphere(dim, center, radius_shell)
    mag_shape_disc = shapes.disc(dim, center, radius_core, height=dim[0])
    mag_shape_core = np.logical_and(mag_shape_sphere, mag_shape_disc)
    mag_shape_shell = np.logical_and(mag_shape_sphere, np.logical_not(mag_shape_core))
    magdata = VectorData(a, mc.create_mag_dist_vortex(mag_shape_shell)) * rate_core_to_shell
    magdata += VectorData(a, mc.create_mag_dist_homog(mag_shape_core, phi=0, theta=0))
    return magdata
