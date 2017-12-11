# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""Convenience function for phase mapping magnetic or charge distributions."""

import logging

from ..kernel import Kernel, KernelCharge
from ..phasemapper import PhaseMapperRDFC, PhaseMapperFDFC, PhaseMapperCharge
from ..projector import RotTiltProjector, XTiltProjector, YTiltProjector, SimpleProjector

__all__ = ['pm']
_log = logging.getLogger(__name__)

# TODO: rename magdata to vecdata everywhere!


def pm(fielddata, mode='z', b_0=1, electrode_vec=(1E6, 1E6), mapper='RDFC', **kwargs):
    """Convenience function for fast electric charge and magnetic phase mapping.

    Parameters
    ----------
    fielddata : :class:`~.VectorData`, or `~.ScalarData`
        A :class:`~.VectorData` or `~.ScalarData` object, from which the projected phase map should be calculated.
    mode: {'z', 'y', 'x', 'x-tilt', 'y-tilt', 'rot-tilt'}, optional
        Projection mode which determines the :class:`~.pyramid.projector.Projector` subclass, which
        is used for the projection. Default is a simple projection along the `z`-direction.
    b_0 : float, optional
        Saturation magnetization in Tesla, which is used for the phase calculation. Default is 1.
    electrode_vec : tuple of float (N=2)
        The norm vector of the counter electrode, (elec_a,elec_b), and the distance to the origin is
        the norm of (elec_a,elec_b). The default value is (1E6, 1E6).
    mapper : :class: '~. PhaseMap'
        A :class: '~. PhaseMap' object, which maps a fielddata into a phase map. The default is 'RDFC'.
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
        projector = RotTiltProjector(fielddata.dim, **kwargs)
    elif mode == 'x-tilt':
        projector = XTiltProjector(fielddata.dim, **kwargs)
    elif mode == 'y-tilt':
        projector = YTiltProjector(fielddata.dim, **kwargs)
    elif mode in ['x', 'y', 'z']:
        projector = SimpleProjector(fielddata.dim, axis=mode, **kwargs)
    else:
        raise ValueError("Invalid mode (use 'x', 'y', 'z', 'x-tilt', 'y-tilt' or 'rot-tilt')")
    # Project:
    field_proj = projector(fielddata)
    # Set up phasemapper and map phase:
    if mapper == 'RDFC':
        phasemapper = PhaseMapperRDFC(Kernel(fielddata.a, projector.dim_uv, b_0=b_0))
    elif mapper == 'FDFC':
        phasemapper = PhaseMapperFDFC(fielddata.a, projector.dim_uv, b_0=b_0, padding=padding)
        # Set up phasemapper and map phase:
    elif mapper == 'Charge':
        phasemapper = PhaseMapperCharge(KernelCharge(fielddata.a, projector.dim_uv, electrode_vec=electrode_vec))
    else:
        raise ValueError("Invalid mapper (use 'RDFC', 'FDFC' or 'Charge'")
    phasemap = phasemapper(field_proj)
    # Get mask from fielddata:
    phasemap.mask = field_proj.get_mask()[0, ...]
    # Return phase:
    return phasemap
