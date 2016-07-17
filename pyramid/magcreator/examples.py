# -*- coding: utf-8 -*-

import logging

import numpy as np

import random as rnd

from . import magcreator as mc
from . import shapes
from ..fielddata import VectorData

_log = logging.getLogger(__name__)


def pyramid_logo(a=1., dim=(1, 256, 256), phi=-np.pi / 2, theta=np.pi / 2):
    """Create pyramid logo."""
    mag_shape = np.zeros(dim)
    x = range(dim[2])
    y = range(dim[1])
    xx, yy = np.meshgrid(x, y)
    bottom = (yy >= 0.25 * dim[1])
    left = (yy <= 0.75 / 0.5 * dim[1] / dim[2] * xx)
    right = np.fliplr(left)
    mag_shape[0, ...] = np.logical_and(np.logical_and(left, right), bottom)
    return VectorData(a, mc.create_mag_dist_homog(mag_shape, phi, theta))


def singularity(a=1., dim=(5, 5, 5), center=None):
    """Create magnetic singularity."""
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    zz, yy, xx = np.indices(dim)
    magnitude = np.array((xx - center[2], yy - center[1], zz - center[0])).astype(float)
    magnitude /= np.sqrt((magnitude ** 2).sum(axis=0))
    return VectorData(a, magnitude)


def homog_pixel(a=1., dim=(1, 9, 9), pixel=None, phi=np.pi / 4, theta=np.pi / 2):
    """Create single magnetised slab."""
    if pixel is None:
        pixel = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    mag_shape = shapes.pixel(dim, pixel)
    return VectorData(a, mc.create_mag_dist_homog(mag_shape, phi, theta))


def homog_slab(a=1., dim=(32, 32, 32), center=None, width=None, phi=np.pi / 4, theta=np.pi / 4):
    """Create homogeneous slab magnetisation distribution."""
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if width is None:
        width = (dim[0] // 8, dim[1] // 2, dim[2] // 4)
    mag_shape = shapes.slab(dim, center, width)
    return VectorData(a, mc.create_mag_dist_homog(mag_shape, phi, theta))


def homog_disc(a=1., dim=(32, 32, 32), center=None, radius=None, height=None,
               phi=np.pi / 4, theta=np.pi / 4):
    """Create homogeneous disc magnetisation distribution."""
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if radius is None:
        radius = dim[2] // 4
    if height is None:
        height = dim[0] // 2
    mag_shape = shapes.disc(dim, center, radius, height)
    return VectorData(a, mc.create_mag_dist_homog(mag_shape, phi, theta))


def homog_sphere(a=1., dim=(32, 32, 32), center=None, radius=None, phi=np.pi / 4, theta=np.pi / 4):
    """Create homogeneous sphere magnetisation distribution."""
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if radius is None:
        radius = dim[2] // 4
    mag_shape = shapes.sphere(dim, center, radius)
    return VectorData(a, mc.create_mag_dist_homog(mag_shape, phi, theta))


def homog_filament(a=1., dim=(1, 21, 21), pos=None, phi=np.pi / 2, theta=np.pi / 2):
    """Create magnetisation distribution of a single magnetised filaments."""
    if pos is None:
        pos = (0, dim[1] // 2)
    mag_shape = shapes.filament(dim, pos)
    return VectorData(a, mc.create_mag_dist_homog(mag_shape, phi, theta))


def homog_alternating_filament(a=1., dim=(1, 21, 21), spacing=5, phi=np.pi / 2, theta=np.pi / 2):
    """Create magnetisation distribution of alternating filaments."""
    count = int((dim[1] - 1) / spacing) + 1
    mag_data = VectorData(a, np.zeros((3,) + dim))
    for i in range(count):
        pos = i * spacing
        mag_shape = shapes.filament(dim, (0, pos))
        mag_data += VectorData(a, mc.create_mag_dist_homog(mag_shape, phi, theta))
        phi *= -1  # Switch the angle
    return mag_data


def homog_array_sphere_disc_slab(a=1., dim=(64, 128, 128), center_sp=(32, 96, 64), radius_sp=24,
                                 center_di=(32, 32, 96), radius_di=24, height_di=24,
                                 center_sl=(32, 32, 32), width_sl=(48, 48, 48)):
    """Create array of several  magnetisation distribution (sphere, disc and slab)."""
    mag_shape_sphere = shapes.sphere(dim, center_sp, radius_sp)
    mag_shape_disc = shapes.disc(dim, center_di, radius_di, height_di)
    mag_shape_slab = shapes.slab(dim, center_sl, width_sl)
    mag_data = VectorData(a, mc.create_mag_dist_homog(mag_shape_sphere, np.pi))
    mag_data += VectorData(a, mc.create_mag_dist_homog(mag_shape_disc, np.pi / 2))
    mag_data += VectorData(a, mc.create_mag_dist_homog(mag_shape_slab, np.pi / 4))
    return mag_data


def homog_random_pixels(a=1., dim=(1, 64, 64), count=10, rnd_seed=24):
    """Create random magnetised pixels."""
    rnd.seed(rnd_seed)
    mag_data = VectorData(a, np.zeros((3,) + dim))
    for i in range(count):
        pixel = (rnd.randrange(dim[0]), rnd.randrange(dim[1]), rnd.randrange(dim[2]))
        mag_shape = shapes.pixel(dim, pixel)
        phi = 2 * np.pi * rnd.random()
        mag_data += VectorData(a, mc.create_mag_dist_homog(mag_shape, phi))
    return mag_data


def homog_random_slabs(a=1., dim=(1, 64, 64), count=10, width_max=5, rnd_seed=2):
    """Create random magnetised slabs."""
    rnd.seed(rnd_seed)
    mag_data = VectorData(a, np.zeros((3,) + dim))
    for i in range(count):
        width = (1, rnd.randint(1, width_max), rnd.randint(1, width_max))
        center = (rnd.randrange(int(width[0] / 2), dim[0] - int(width[0] / 2)),
                  rnd.randrange(int(width[1] / 2), dim[1] - int(width[1] / 2)),
                  rnd.randrange(int(width[2] / 2), dim[2] - int(width[2] / 2)))
        mag_shape = shapes.slab(dim, center, width)
        phi = 2 * np.pi * rnd.random()
        mag_data += VectorData(a, mc.create_mag_dist_homog(mag_shape, phi))
    return mag_data


def vortex_slab(a=1., dim=(32, 32, 32), center=None, width=None, axis='z'):
    """Create vortex slab magnetisation distribution."""
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if width is None:
        width = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    mag_shape = shapes.slab(dim, center, width)
    magnitude = mc.create_mag_dist_vortex(mag_shape, center, axis)
    return VectorData(a, magnitude)


def vortex_disc(a=1., dim=(32, 32, 32), center=None, radius=None, height=None, axis='z'):
    """Create vortex disc magnetisation distribution."""
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if radius is None:
        radius = dim[2] // 4
    if height is None:
        height = dim[0] // 2
    mag_shape = shapes.disc(dim, center, radius, height, axis)
    magnitude = mc.create_mag_dist_vortex(mag_shape, center, axis)
    return VectorData(a, magnitude)


def vortex_alternating_discs(a=1., dim=(80, 32, 32), count=8):
    """Create pillar of alternating vortex disc magnetisation distributions."""
    segment_height = dim[0] // (count + 2)
    mag_data = VectorData(a, np.zeros((3,) + dim))
    for i in range(count):
        axis = 'z' if i % 2 == 0 else '-z'
        center = (segment_height * (i + 1 + 0.5), dim[1] // 2, dim[2] // 2)
        radius = dim[2] // 4
        height = segment_height
        mag_shape = shapes.disc(dim, center=center, radius=radius, height=height)
        mag_amp = mc.create_mag_dist_vortex(mag_shape=mag_shape, center=center, axis=axis)
        mag_data += VectorData(1, mag_amp)
    return mag_data


def vortex_sphere(a=1., dim=(32, 32, 32), center=None, radius=None, axis = 'z'):
    """Create vortex sphere magnetisation distribution."""
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
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if radius_core is None:
        radius_core = dim[1] // 8
    if radius_shell is None:
        radius_shell = dim[1] // 4
    if height is None:
        height = dim[0] // 2
    mag_shape_core = shapes.disc(dim, center, radius_core, height)
    mag_shape_outer = shapes.disc(dim, center, radius_shell, height)
    mag_shape_horseshoe = np.logical_xor(mag_shape_outer, mag_shape_core)
    mag_shape_horseshoe[:, dim[1] // 2:, :] = False
    return VectorData(a, mc.create_mag_dist_vortex(mag_shape_horseshoe))


def core_shell_disc(a=1., dim=(32, 32, 32), center=None, radius_core=None,
                    radius_shell=None, height=None, rate_core_to_shell=0.75):
    """Create magnetic core shell disc magnetisation distribution."""
    if center is None:
        center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
    if radius_core is None:
        radius_core = dim[1] // 8
    if radius_shell is None:
        radius_shell = dim[1] // 4
    if height is None:
        height = dim[0] // 2
    mag_shape_core = shapes.disc(dim, center, radius_core, height)
    mag_shape_outer = shapes.disc(dim, center, radius_shell, height)
    mag_shape_shell = np.logical_xor(mag_shape_outer, mag_shape_core)
    mag_data = VectorData(a, mc.create_mag_dist_vortex(mag_shape_shell)) * rate_core_to_shell
    mag_data += VectorData(a, mc.create_mag_dist_homog(mag_shape_core, phi=0, theta=0))
    return mag_data


def core_shell_sphere(a=1., dim=(32, 32, 32), center=None, radius_core=None,
                      radius_shell=None, rate_core_to_shell=0.75):
    """Create magnetic core shell sphere magnetisation distribution."""
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
    mag_data = VectorData(a, mc.create_mag_dist_vortex(mag_shape_shell)) * rate_core_to_shell
    mag_data += VectorData(a, mc.create_mag_dist_homog(mag_shape_core, phi=0, theta=0))
    return mag_data
