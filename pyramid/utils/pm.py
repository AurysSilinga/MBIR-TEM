# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
import logging

from ..kernel import Kernel
from ..phasemapper import PhaseMapperRDFC
from ..projector import RotTiltProjector, XTiltProjector, YTiltProjector, SimpleProjector


__all__ = ['pm']
_log = logging.getLogger(__name__)


def pm(mag_data, mode='z', b_0=1, **kwargs):
    """Convenience function for fast magnetic phase mapping.

    Parameters
    ----------
    mag_data : :class:`~.VectorData`
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
    phase_map : :class:`~pyramid.phasemap.PhaseMap`
        The calculated phase map as a :class:`~.PhaseMap` object.

    """
    _log.debug('Calling pm')
    # Determine projection mode:
    if mode == 'rot-tilt':
        projector = RotTiltProjector(mag_data.dim, **kwargs)
    elif mode == 'x-tilt':
        projector = XTiltProjector(mag_data.dim, **kwargs)
    elif mode == 'y-tilt':
        projector = YTiltProjector(mag_data.dim, **kwargs)
    elif mode in ['x', 'y', 'z']:
        projector = SimpleProjector(mag_data.dim, axis=mode, **kwargs)
    else:
        raise ValueError("Invalid mode (use 'x', 'y', 'z', 'x-tilt', 'y-tilt' or 'rot-tilt')")
    # Project:
    mag_proj = projector(mag_data)
    # Set up phasemapper and map phase:
    phasemapper = PhaseMapperRDFC(Kernel(mag_data.a, projector.dim_uv, b_0=b_0))
    phase_map = phasemapper(mag_proj)
    # Get mask from magdata:
    phase_map.mask = mag_proj.get_mask()[0, ...]
    # Return phase:
    return phase_map
