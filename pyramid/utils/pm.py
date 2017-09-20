# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""Convenience function for phase mapping magnetic distributions."""

import logging

from ..kernel import Kernel
from ..phasemapper import PhaseMapperRDFC, PhaseMapperFDFC
from ..projector import RotTiltProjector, XTiltProjector, YTiltProjector, SimpleProjector

__all__ = ['pm']
_log = logging.getLogger(__name__)

# TODO: rename magdata to vecdata everywhere!

def pm(magdata, mode='z', b_0=1, mapper='RDFC', **kwargs):
    """Convenience function for fast magnetic phase mapping.

    Parameters
    ----------
    magdata : :class:`~.VectorData`
        A :class:`~.VectorData` object, from which the projected phase map should be calculated.
    mode: {'z', 'y', 'x', 'x-tilt', 'y-tilt', 'rot-tilt'}, optional
        Projection mode which determines the :class:`~.pyramid.projector.Projector` subclass, which
        is used for the projection. Default is a simple projection along the `z`-direction.
    b_0 : float, optional
        Saturation magnetization in Tesla, which is used for the phase calculation. Default is 1.
    **kwargs : additional arguments
        Additional arguments like `dim_uv`, 'tilt' or 'rotation', which are passed to the
        projector-constructor, defined by the `mode`.

    Returns
    -------
    phasemap : :class:`~pyramid.phasemap.PhaseMap`
        The calculated phase map as a :class:`~.PhaseMap` object.

    """
    _log.debug('Calling pm')
    # In case of FDFC:
    padding = kwargs.pop('padding', 0)
    # Determine projection mode:
    if mode == 'rot-tilt':
        projector = RotTiltProjector(magdata.dim, **kwargs)
    elif mode == 'x-tilt':
        projector = XTiltProjector(magdata.dim, **kwargs)
    elif mode == 'y-tilt':
        projector = YTiltProjector(magdata.dim, **kwargs)
    elif mode in ['x', 'y', 'z']:
        projector = SimpleProjector(magdata.dim, axis=mode, **kwargs)
    else:
        raise ValueError("Invalid mode (use 'x', 'y', 'z', 'x-tilt', 'y-tilt' or 'rot-tilt')")
    # Project:
    mag_proj = projector(magdata)
    # Set up phasemapper and map phase:
    if mapper == 'RDFC':
        phasemapper = PhaseMapperRDFC(Kernel(magdata.a, projector.dim_uv, b_0=b_0))
    elif mapper == 'FDFC':
        phasemapper = PhaseMapperFDFC(magdata.a, projector.dim_uv, b_0=b_0, padding=padding)
    else:
        raise ValueError("Invalid mapper (use 'RDFC' or 'FDFC'")
    phasemap = phasemapper(mag_proj)
    # Get mask from magdata:
    phasemap.mask = mag_proj.get_mask()[0, ...]
    # Return phase:
    return phasemap
