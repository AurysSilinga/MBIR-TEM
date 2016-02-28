# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""Create phase maps for magnetic distributions with analytic solutions.

This module provides methods for the calculation of the magnetic phase for simple geometries for
which the analytic solutions are known. These can be used for comparison with the phase
calculated by the functions from the :mod:`~pyramid.phasemapper` module.

"""

import logging

import numpy as np
from numpy import pi

from pyramid.phasemap import PhaseMap

__all__ = ['phase_mag_slab', 'phase_mag_slab', 'phase_mag_sphere', 'phase_mag_vortex']
_log = logging.getLogger(__name__)

PHI_0 = 2067.83  # magnetic flux in T*nm²


def phase_mag_slab(dim, a, phi, center, width, b_0=1):
    """Calculate the analytic magnetic phase for a homogeneously magnetized slab.

    Parameters
    ----------
    dim : tuple (N=3)
        The dimensions of the grid `(z, y, x)`.
    a : float
        The grid spacing in nm.
    phi : float
        The azimuthal angle, describing the direction of the magnetization.
    center : tuple (N=3)
        The center of the slab in pixel coordinates `(z, y, x)`.
    width : tuple (N=3)
        The width of the slab in pixel coordinates `(z, y, x)`.
    b_0 : float, optional
        The magnetic induction corresponding to a magnetization `M`\ :sub:`0` in T.
        The default is 1.

    Returns
    -------
    phase_map : :class:`~numpy.ndarray` (N=2)
        The phase as a 2-dimensional array.

    """
    _log.debug('Calling phase_mag_slab')

    # Function for the phase:
    def _phi_mag(x, y):
        def _F_0(x, y):
            A = np.log(x ** 2 + y ** 2 + 1E-30)
            B = np.arctan(x / (y + 1E-30))
            return x * A - 2 * x + 2 * y * B

        return coeff * Lz * (- np.cos(phi) * (_F_0(x - x0 - Lx / 2, y - y0 - Ly / 2) -
                                              _F_0(x - x0 + Lx / 2, y - y0 - Ly / 2) -
                                              _F_0(x - x0 - Lx / 2, y - y0 + Ly / 2) +
                                              _F_0(x - x0 + Lx / 2, y - y0 + Ly / 2))
                             + np.sin(phi) * (_F_0(y - y0 - Ly / 2, x - x0 - Lx / 2) -
                                              _F_0(y - y0 + Ly / 2, x - x0 - Lx / 2) -
                                              _F_0(y - y0 - Ly / 2, x - x0 + Lx / 2) +
                                              _F_0(y - y0 + Ly / 2, x - x0 + Lx / 2)))

    # Process input parameters:
    z_dim, y_dim, x_dim = dim
    y0 = a * center[1]  # y0, x0 define the center of a pixel,
    x0 = a * center[2]  # hence: (cellindex + 0.5) * grid spacing
    Lz, Ly, Lx = a * width[0], a * width[1], a * width[2]
    coeff = - b_0 / (4 * PHI_0)  # Minus because of negative z-direction
    # Create grid:
    x = np.linspace(a / 2, x_dim * a - a / 2, num=x_dim)
    y = np.linspace(a / 2, y_dim * a - a / 2, num=y_dim)
    xx, yy = np.meshgrid(x, y)
    # Return phase:
    return PhaseMap(a, _phi_mag(xx, yy))


def phase_mag_disc(dim, a, phi, center, radius, height, b_0=1):
    """Calculate the analytic magnetic phase for a homogeneously magnetized disc.

    Parameters
    ----------
    dim : tuple (N=3)
        The dimensions of the grid `(z, y, x)`.
    a : float
        The grid spacing in nm.
    phi : float
        The azimuthal angle, describing the direction of the magnetization.
    center : tuple (N=3)
        The center of the disc in pixel coordinates `(z, y, x)`.
    radius : float
        The radius of the disc in pixel coordinates.
    height : float
        The height of the disc in pixel coordinates.
    b_0 : float, optional
        The magnetic induction corresponding to a magnetization `M`\ :sub:`0` in T.
        The default is 1.

    Returns
    -------
    phase_map : :class:`~numpy.ndarray` (N=2)
        The phase as a 2-dimensional array.

    """
    _log.debug('Calling phase_mag_disc')

    # Function for the phase:
    def _phi_mag(x, y):
        r = np.hypot(x - x0, y - y0)
        result = coeff * Lz * ((y - y0) * np.cos(phi) - (x - x0) * np.sin(phi))
        result *= np.where(r <= R, 1, (R / (r + 1E-30)) ** 2)
        return result

    # Process input parameters:
    z_dim, y_dim, x_dim = dim
    y0 = a * center[1]
    x0 = a * center[2]
    Lz = a * height
    R = a * radius
    coeff = pi * b_0 / (2 * PHI_0)  # Minus is gone because of negative z-direction
    # Create grid:
    x = np.linspace(a / 2, x_dim * a - a / 2, num=x_dim)
    y = np.linspace(a / 2, y_dim * a - a / 2, num=y_dim)
    xx, yy = np.meshgrid(x, y)
    # Return phase:
    return PhaseMap(a, _phi_mag(xx, yy))


def phase_mag_sphere(dim, a, phi, center, radius, b_0=1):
    """Calculate the analytic magnetic phase for a homogeneously magnetized sphere.

    Parameters
    ----------
    dim : tuple (N=3)
        The dimensions of the grid `(z, y, x)`.
    a : float
        The grid spacing in nm.
    phi : float
        The azimuthal angle, describing the direction of the magnetization.
    center : tuple (N=3)
        The center of the sphere in pixel coordinates `(z, y, x)`.
    radius : float
        The radius of the sphere in pixel coordinates.
    b_0 : float, optional
        The magnetic induction corresponding to a magnetization `M`\ :sub:`0` in T.
        The default is 1.

    Returns
    -------
    phase_map : :class:`~numpy.ndarray` (N=2)
        The phase as a 2-dimensional array.

    """
    _log.debug('Calling phase_mag_sphere')

    # Function for the phase:
    def _phi_mag(x, y):
        r = np.hypot(x - x0, y - y0)
        result = coeff * R ** 3 / (r + 1E-30) ** 2 * (
            (y - y0) * np.cos(phi) - (x - x0) * np.sin(phi))
        result *= np.where(r > R, 1, (1 - (1 - (r / R) ** 2) ** (3. / 2.)))
        return result

    # Process input parameters:
    z_dim, y_dim, x_dim = dim
    y0 = a * center[1]
    x0 = a * center[2]
    R = a * radius
    coeff = 2. / 3. * pi * b_0 / PHI_0  # Minus is gone because of negative z-direction
    # Create grid:
    x = np.linspace(a / 2, x_dim * a - a / 2, num=x_dim)
    y = np.linspace(a / 2, y_dim * a - a / 2, num=y_dim)
    xx, yy = np.meshgrid(x, y)
    # Return phase:
    return PhaseMap(a, _phi_mag(xx, yy))


def phase_mag_vortex(dim, a, center, radius, height, b_0=1):
    """Calculate the analytic magnetic phase for a vortex state disc.

    Parameters
    ----------
    dim : tuple (N=3)
        The dimensions of the grid `(z, y, x)`.
    a : float
        The grid spacing in nm.
    center : tuple (N=3)
        The center of the disc in pixel coordinates `(z, y, x)`, which is also the vortex center.
    radius : float
        The radius of the disc in pixel coordinates.
    height : float
        The height of the disc in pixel coordinates.
    b_0 : float, optional
        The magnetic induction corresponding to a magnetization `M`\ :sub:`0` in T.
        The default is 1.

    Returns
    -------
    phase_map : :class:`~numpy.ndarray` (N=2)
        The phase as a 2-dimensional array.

    """
    _log.debug('Calling phase_mag_vortex')

    # Function for the phase:
    def _phi_mag(x, y):
        r = np.hypot(x - x0, y - y0)
        result = coeff * np.where(r <= R, r - R, 0)
        return result

    # Process input parameters:
    z_dim, y_dim, x_dim = dim
    y0 = a * center[1]
    x0 = a * center[2]
    Lz = a * height
    R = a * radius
    coeff = - pi * b_0 * Lz / PHI_0  # Minus because of negative z-direction
    # Create grid:
    x = np.linspace(a / 2, x_dim * a - a / 2, num=x_dim)
    y = np.linspace(a / 2, y_dim * a - a / 2, num=y_dim)
    xx, yy = np.meshgrid(x, y)
    # Return phase:
    return PhaseMap(a, _phi_mag(xx, yy))
