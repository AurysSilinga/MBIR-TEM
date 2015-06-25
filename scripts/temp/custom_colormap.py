# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 12:14:28 2015

@author: Jan
"""

from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import logging


__all__ = ['create_custom_colormap']
_log = logging.getLogger(__name__)


def create_custom_colormap(levels=15, N=256):

    '''Class for storing phase map data.

    Represents 2-dimensional phase maps. The phase information itself is stored as a 2-dimensional
    matrix in `phase`, but can also be accessed as a vector via `phase_vec`. :class:`~.PhaseMap`
    objects support negation, arithmetic operators (``+``, ``-``, ``*``) and their augmented
    counterparts (``+=``, ``-=``, ``*=``), with numbers and other :class:`~.PhaseMap`
    objects, if their dimensions and grid spacings match. It is possible to load data from NetCDF4
    or textfiles or to save the data in these formats. Methods for plotting the phase or a
    corresponding holographic contour map are provided. Holographic contour maps are created by
    taking the cosine of the (optionally amplified) phase and encoding the direction of the
    2-dimensional gradient via color. The directional encoding can be seen by using the
    :func:`~.make_color_wheel` function. Use the :func:`~.display_combined` function to plot the
    phase map and the holographic contour map next to each other.

    Attributes
    ----------
    a: float
        The grid spacing in nm.
    dim_uv: tuple (N=2)
        Dimensions of the grid.
    phase: :class:`~numpy.ndarray` (N=2)
        Matrix containing the phase shift.
    phase_vec: :class:`~numpy.ndarray` (N=2)
        Vector containing the phase shift.
    mask: :class:`~numpy.ndarray` (boolean, N=2, optional)
        Mask which determines the projected magnetization distribution, gotten from MIP images or
        otherwise acquired. Defaults to an array of ones (all pixels are considered).
    confidence: :class:`~numpy.ndarray` (N=2, optional)
        Confidence array which determines the trust of specific regions of the phase_map. A value
        of 1 means the pixel is trustworthy, a value of 0 means it is not. Defaults to an array of
        ones (full trust for all pixels). Can be used for the construction of Se_inv.
    unit: {'rad', 'mrad'}, optional
        Set the unit of the phase map. This is important for the :func:`display` function,
        because the phase is scaled accordingly. Does not change the phase itself, which is
        always in `rad`.

    '''  # TODO: Docstring!

    CDICT = {'red': [(0.00, 1.0, 0.0),
                     (0.25, 1.0, 1.0),
                     (0.50, 1.0, 1.0),
                     (0.75, 0.0, 0.0),
                     (1.00, 0.0, 1.0)],

             'green': [(0.00, 0.0, 0.0),
                       (0.25, 0.0, 0.0),
                       (0.50, 1.0, 1.0),
                       (0.75, 1.0, 1.0),
                       (1.00, 0.0, 1.0)],

             'blue': [(0.00, 1.0, 1.0),
                      (0.25, 0.0, 0.0),
                      (0.50, 0.0, 0.0),
                      (0.75, 0.0, 0.0),
                      (1.00, 1.0, 1.0)]}

    CDICT_INV = {'red': [(0.00, 0.0, 1.0),
                         (0.25, 0.0, 0.0),
                         (0.50, 0.0, 0.0),
                         (0.75, 1.0, 1.0),
                         (1.00, 1.0, 0.0)],

                 'green': [(0.00, 1.0, 1.0),
                           (0.25, 1.0, 1.0),
                           (0.50, 0.0, 0.0),
                           (0.75, 0.0, 0.0),
                           (1.00, 1.0, 0.0)],

                 'blue': [(0.00, 0.0, 0.0),
                          (0.25, 1.0, 1.0),
                          (0.50, 1.0, 1.0),
                          (0.75, 1.0, 1.0),
                          (1.00, 0.0, 0.0)]}

    r, g, b = [], [], []
    center = levels//2
    pos_sat = np.ones(levels)
    pos_sat[0:center] = [i/center for i in range(center)]
    neg_sat = np.zeros(levels)
    neg_sat[center+1:] = [(i+1)/center for i in range(center)]
    print pos_sat
    print neg_sat
    # example for 5 levels (from black to color to white):
    # [ 0.   0.5  1.   1.   1. ]
    # [ 0.   0.   0.   0.5  1. ]

    for i in range(levels):
        print i + 1
        inter_points = np.linspace(i/levels, (i+1)/levels, 5)  # interval points
        print inter_points

        CDICT = {'red': [(0.00, 1.0, 0.0),
                         (0.25, 1.0, 1.0),
                         (0.50, 1.0, 1.0),
                         (0.75, 0.0, 0.0),
                         (1.00, 0.0, 1.0)],

             'green': [(0.00, 0.0, 0.0),
                       (0.25, 0.0, 0.0),
                       (0.50, 1.0, 1.0),
                       (0.75, 1.0, 1.0),
                       (1.00, 0.0, 1.0)],

             'blue': [(0.00, 1.0, 1.0),
                      (0.25, 0.0, 0.0),
                      (0.50, 0.0, 0.0),
                      (0.75, 0.0, 0.0),
                      (1.00, 1.0, 1.0)]}

        r.append((inter_points[0], 0, neg_sat[i]))
        r.append((inter_points[1], pos_sat[i], pos_sat[i]))
        r.append((inter_points[2], pos_sat[i], pos_sat[i]))
        r.append((inter_points[3], neg_sat[i], neg_sat[i]))
        r.append((inter_points[4], neg_sat[i], 0))
        print r

        g.append((inter_points[0], 0, neg_sat[i]))
        g.append((inter_points[1], neg_sat[i], neg_sat[i]))
        g.append((inter_points[2], pos_sat[i], pos_sat[i]))
        g.append((inter_points[3], pos_sat[i], pos_sat[i]))
        g.append((inter_points[4], neg_sat[i], 0))
        print g

        b.append((inter_points[0], 0, pos_sat[i]))
        b.append((inter_points[1], neg_sat[i], neg_sat[i]))
        b.append((inter_points[2], neg_sat[i], neg_sat[i]))
        b.append((inter_points[3], neg_sat[i], neg_sat[i]))
        b.append((inter_points[4], pos_sat[i], 0))
        print b

#        r.append((inter_points[0], 0, 0))
#        r.append((inter_points[1], pos_sat[i], pos_sat[i]))
#
#        g.append((inter_points[0], 0, 0))
#        g.append((inter_points[1], neg_sat[i], neg_sat[i]))
#
#        b.append((inter_points[0], 0, 0))
#        b.append((inter_points[1], neg_sat[i], neg_sat[i]))

    cdict = {'red': r, 'green': g, 'blue': b}
    return mpl.colors.LinearSegmentedColormap('custom_colormap', cdict, N)


x = np.arange(0, np.pi, 0.1)
y = np.arange(0, 2*np.pi, 0.1)
X, Y = np.meshgrid(x, y)
Z = np.cos(X) * np.sin(Y) * 10

fig = plt.figure(figsize=(7, 7))
axis = fig.add_subplot(1, 1, 1)
im = axis.imshow(Z, cmap=create_custom_colormap(levels=3, N=3))
fig.colorbar(im)
